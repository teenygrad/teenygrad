"""Ground truth for one Inkling-2 layer, from transformers' own modules driven with the
real checkpoint weights. Experts are dequantised only for the routed slots."""
import json, struct, sys, glob
import numpy as np, torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

DIR = "/home/petr/models/Inkling-2-Mini"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOKEN = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ref_layer0.bin"

# ---- checkpoint access ----------------------------------------------------
HDR = {}
for f in glob.glob(f"{DIR}/*.safetensors"):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
        base = 8 + n
    for k, v in h.items():
        if k != "__metadata__":
            HDR[k] = (f, base + v["data_offsets"][0], base + v["data_offsets"][1],
                      v["shape"], v["dtype"])

DT = {"F32": np.float32, "BF16": None, "U8": np.uint8, "F8_E4M3": np.uint8}

def raw(name):
    f, a, b, shape, dt = HDR[name]
    buf = np.fromfile(f, dtype=np.uint8, count=b - a, offset=a)
    if dt == "BF16":
        u16 = buf.view(np.uint16).astype(np.uint32) << 16
        return u16.view(np.float32).reshape(shape)
    return buf.view(DT[dt]).reshape(shape)

def t(name):
    return torch.from_numpy(np.ascontiguousarray(raw(name))).float()

P = f"model.llm.layers.{LAYER}."

# ---- NVFP4 / MXFP8 dequant ------------------------------------------------
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

def e4m3(b):
    b = b.astype(np.uint32)
    sgn = np.where(b >> 7 == 1, -1.0, 1.0).astype(np.float32)
    e = (b >> 3) & 0xF
    m = (b & 7).astype(np.float32)
    v = np.where(e == 0, m * 2.0 ** -9, (1.0 + m / 8.0) * 2.0 ** (e.astype(np.float32) - 7.0))
    return (sgn * v).astype(np.float32)

def dequant_nvfp4(rows_bytes, rows_scale, s2):
    """rows_bytes [R, K/2] uint8, rows_scale [R, K/16] uint8 -> [R, K] f32."""
    lo = rows_bytes & 0xF
    hi = rows_bytes >> 4
    nib = np.empty((rows_bytes.shape[0], rows_bytes.shape[1] * 2), dtype=np.uint8)
    nib[:, 0::2] = lo
    nib[:, 1::2] = hi
    sgn = np.where(nib >> 3 == 1, -1.0, 1.0).astype(np.float32)
    mag = E2M1[(nib & 7).astype(np.int32)]
    vals = sgn * mag
    sc = e4m3(rows_scale)                       # [R, K/16]
    vals *= np.repeat(sc, 16, axis=1)
    return vals * s2

def dequant_mxfp8(rows_bytes, rows_scale):
    """rows_bytes [R, K] uint8 E4M3, rows_scale [R, K/32] uint8 E8M0."""
    vals = e4m3(rows_bytes)
    e = rows_scale.astype(np.int32)
    sc = np.where(e == 0, 2.0 ** -127, 2.0 ** (e.astype(np.float32) - 127.0)).astype(np.float32)
    return vals * np.repeat(sc, 32, axis=1)

NVFP4 = LAYER < 20

def expert_rows(which, expert):
    """Dequantised [rows, K] for one expert of `w13_weight` or `w2_weight`."""
    b = raw(P + f"mlp.experts.{which}_weight")[expert]
    s = raw(P + f"mlp.experts.{which}_weight_scale")[expert]
    if NVFP4:
        amax = raw(P + f"mlp.experts.{which}_weight_amax")[expert]
        den = raw(P + f"mlp.experts.{which}_weight_outer_scale_denominator")[0]
        return dequant_nvfp4(b, s, np.float32(amax / den))
    return dequant_mxfp8(b, s)

class LazyExperts(torch.nn.Module):
    """Inkling2LoopExperts, dequantising only the experts a token actually routes to."""
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg.n_routed_experts
        self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(cfg)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        for pos in range(top_k_index.shape[0]):
            for slot in range(top_k_index.shape[1]):
                e = int(top_k_index[pos, slot])
                w13 = torch.from_numpy(expert_rows("w13", e))   # [2I, H]
                w2 = torch.from_numpy(expert_rows("w2", e))     # [H, I]
                x = hidden_states[pos:pos + 1].float()
                gu = torch.nn.functional.linear(x, w13)
                g, u = gu.chunk(2, dim=-1)
                h = self.act_fn(g, u)
                y = torch.nn.functional.linear(h, w2)
                out[pos:pos + 1] += y * float(top_k_weights[pos, slot])
        return out

# ---- build the layer ------------------------------------------------------
cfg_json = json.load(open(f"{DIR}/config.json"))["text_config"]
cfg = Inkling2LoopTextConfig(**cfg_json)
is_sliding = cfg.layer_types[LAYER] == "hybrid_sliding"
cls = M.Inkling2LoopSlidingDecoderLayer if is_sliding else M.Inkling2LoopFullDecoderLayer
layer = cls(cfg, LAYER).eval()
layer.mlp.experts = LazyExperts(cfg)

sd = {
    "input_layernorm.weight": t(P + "attn_norm.weight"),
    "post_attention_layernorm.weight": t(P + "mlp_norm.weight"),
    "post_attention_convolution.conv1d.weight": t(P + "attn_sconv.weight"),
    "post_mlp_convolution.conv1d.weight": t(P + "mlp_sconv.weight"),
    "self_attn.q_proj.linear.weight": t(P + "attn.wq_du.weight"),
    "self_attn.q_proj.norm.scale": t(P + "attn.q_l2_clip.scale"),
    "self_attn.o_proj.gate_proj.weight": t(P + "attn.gate_attn.weight"),
    "self_attn.o_proj.linear.weight": t(P + "attn.wo_ud.weight"),
    "self_attn.sink_proj.weight": t(P + "attn.attn_sink_proj.weight"),
    "self_attn.relative_position_bias.linear.weight": t(P + "attn.wr_du.weight"),
    "self_attn.relative_position_bias.norm.scale": t(P + "attn.rel_l2_clip.scale"),
    "self_attn.relative_position_bias.proj": t(P + "attn.rel_logits_proj.proj"),
    "self_attn.k_proj.linear.weight": t(P + "attn.wk_dv.weight"),
    "self_attn.v_proj.linear.weight": t(P + "attn.wv_dv.weight"),
    "mlp.gate.weight": t(P + "mlp.gate.weight"),
    "mlp.gate.e_score_correction_bias": t(P + "mlp.gate.bias"),
    "mlp_input.query": t(f"model.llm.attn_res.mlp_queries.{LAYER}"),
}
if is_sliding:
    sd["self_attn.k_proj.sconv.conv1d.weight"] = t(P + "attn.k_sconv.weight")
    sd["self_attn.v_proj.sconv.conv1d.weight"] = t(P + "attn.v_sconv.weight")
else:
    sd["self_attn.k_proj.sconv.static_weight"] = t(P + "attn.k_sconv.weight").squeeze(1).T
    sd["self_attn.v_proj.sconv.static_weight"] = t(P + "attn.v_sconv.weight").squeeze(1).T
    sd["self_attn.k_proj.sconv.weight_proj.weight"] = t(P + "attn.k_sconv.weight_proj")
    sd["self_attn.v_proj.sconv.weight_proj.weight"] = t(P + "attn.v_sconv.weight_proj")
if LAYER > 0:
    q = t(f"model.llm.attn_res.attn_queries.{LAYER - 1}")
    sd["attention_input.query"] = q if q.dim() == 1 else q[0]

missing, unexpected = layer.load_state_dict(sd, strict=False)
real_missing = [m for m in missing if "experts" not in m]
print("missing:", real_missing)
print("unexpected:", unexpected)

# ---- the embedding, as the model computes it ------------------------------
emb = raw("model.llm.embed.weight")[TOKEN].astype(np.float32)
enorm = raw("model.llm.embed_norm.weight").astype(np.float32)
rms = np.sqrt((emb * emb).mean() + cfg.rms_norm_eps)
x = torch.from_numpy((emb / rms) * enorm).view(1, 1, -1).float()

taps = {}
def tap(name):
    def hook(_m, _i, o):
        v = o[0] if isinstance(o, tuple) else o
        taps[name] = v.detach().reshape(-1).float().numpy().copy()
    return hook
layer.self_attn.register_forward_hook(tap("attn"))
layer.self_attn.q_proj.register_forward_hook(tap("q"))
layer.self_attn.k_proj.register_forward_hook(tap("k"))
layer.self_attn.v_proj.register_forward_hook(tap("v"))
layer.self_attn.sink_proj.register_forward_hook(tap("sinks"))
layer.self_attn.relative_position_bias.register_forward_hook(tap("bias"))
layer.input_layernorm.register_forward_hook(tap("attn_norm"))
layer.post_attention_convolution.register_forward_hook(tap("attn_sconv"))
layer.mlp.register_forward_hook(tap("moe"))
layer.post_attention_layernorm.register_forward_hook(tap("mlp_norm"))
layer.mlp_input.register_forward_hook(tap("mlp_input"))

block_states = x.new_zeros(1, 1, 0, x.shape[-1])
rel = M.get_relative_distance(x, None, is_sliding=is_sliding)
pos = torch.arange(1).unsqueeze(0)
with torch.no_grad():
    out = layer(x, block_states, attention_mask=None, relative_distance=rel,
                conv_mask=None, cache=None, position_ids=pos)
y = out[0] if isinstance(out, tuple) else out
y = y.reshape(-1).float().numpy().astype("<f4")
y.tofile(OUT)
print("layer", LAYER, "token", TOKEN, "-> absmax", float(np.abs(y).max()),
      "mean", float(y.mean()), "first4", y[:4])
print("wrote", OUT, y.size, "floats")
for k in ("attn_norm", "q", "k", "v", "sinks", "bias", "attn", "attn_sconv", "mlp_input", "mlp_norm", "moe"):
    v = taps.get(k)
    if v is None:
        continue
    v.astype("<f4").tofile(OUT.replace(".bin", f".{k}.bin"))
    print(f"  {k:10s} absmax {abs(v).max():.4f} mean {v.mean():+.6f} first3 {v[:3]}")
