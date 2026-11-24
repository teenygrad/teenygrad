"""Ground truth for one Inkling-2 layer, from transformers' own modules driven with the
real checkpoint weights. Experts are dequantised only for the routed slots."""
import json, struct, sys, glob
import numpy as np, torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

DIR = "/home/petr/models/Inkling-2-Mini"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
TOKEN = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ref_full.bin"
ORDER = __import__("os").environ.get("INKLING2_MSG_ORDER", "qkvm")

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

def expert_rows(prefix, nvfp4, which, expert):
    """Dequantised [rows, K] for one expert of `w13_weight` or `w2_weight`.

    The prefix is passed in, not read from a global: building every layer up front
    and then reading a global set by the last `build` call makes every layer use the
    last layer's experts.
    """
    b = raw(prefix + f"mlp.experts.{which}_weight")[expert]
    s = raw(prefix + f"mlp.experts.{which}_weight_scale")[expert]
    if nvfp4:
        amax = raw(prefix + f"mlp.experts.{which}_weight_amax")[expert]
        den = raw(prefix + f"mlp.experts.{which}_weight_outer_scale_denominator")[0]
        return dequant_nvfp4(b, s, np.float32(amax / den))
    return dequant_mxfp8(b, s)

class LazyExperts(torch.nn.Module):
    """Inkling2LoopExperts, dequantising only the experts a token actually routes to."""
    def __init__(self, cfg, prefix, nvfp4):
        super().__init__()
        self.num_experts = cfg.n_routed_experts
        self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(cfg)
        self.prefix, self.nvfp4 = prefix, nvfp4

    def forward(self, hidden_states, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        for pos in range(top_k_index.shape[0]):
            for slot in range(top_k_index.shape[1]):
                e = int(top_k_index[pos, slot])
                w13 = torch.from_numpy(expert_rows(self.prefix, self.nvfp4, "w13", e))
                w2 = torch.from_numpy(expert_rows(self.prefix, self.nvfp4, "w2", e))
                x = hidden_states[pos:pos + 1].float()
                gu = torch.nn.functional.linear(x, w13)
                g, u = gu.chunk(2, dim=-1)
                h = self.act_fn(g, u)
                y = torch.nn.functional.linear(h, w2)
                out[pos:pos + 1] += y * float(top_k_weights[pos, slot])
        return out

# ---- build the layers -----------------------------------------------------
cfg_json = json.load(open(f"{DIR}/config.json"))["text_config"]
cfg = Inkling2LoopTextConfig(**cfg_json)

def state_dict(idx, sliding):
    p = f"model.llm.layers.{idx}."
    sd = {
        "post_attention_layernorm.weight": t(p + "mlp_norm.weight"),
        "post_attention_convolution.conv1d.weight": t(p + "attn_sconv.weight"),
        "post_mlp_convolution.conv1d.weight": t(p + "mlp_sconv.weight"),
        "self_attn.q_proj.linear.weight": t(p + "attn.wq_du.weight"),
        "self_attn.q_proj.norm.scale": t(p + "attn.q_l2_clip.scale"),
        "self_attn.o_proj.gate_proj.weight": t(p + "attn.gate_attn.weight"),
        "self_attn.o_proj.linear.weight": t(p + "attn.wo_ud.weight"),
        "self_attn.sink_proj.weight": t(p + "attn.attn_sink_proj.weight"),
        "self_attn.relative_position_bias.linear.weight": t(p + "attn.wr_du.weight"),
        "self_attn.relative_position_bias.norm.scale": t(p + "attn.rel_l2_clip.scale"),
        "self_attn.relative_position_bias.proj": t(p + "attn.rel_logits_proj.proj"),
        "self_attn.k_proj.linear.weight": t(p + "attn.wk_dv.weight"),
        "self_attn.v_proj.linear.weight": t(p + "attn.wv_dv.weight"),
        "mlp.gate.weight": t(p + "mlp.gate.weight"),
        "mlp.gate.e_score_correction_bias": t(p + "mlp.gate.bias"),
        "mlp_input.query": t(f"model.llm.attn_res.mlp_queries.{idx}"),
    }
    if sliding:
        sd["input_layernorm.weight"] = t(p + "attn_norm.weight")
        sd["self_attn.k_proj.sconv.conv1d.weight"] = t(p + "attn.k_sconv.weight")
        sd["self_attn.v_proj.sconv.conv1d.weight"] = t(p + "attn.v_sconv.weight")
    else:
        # A full-attention layer norms the query, key and value inputs separately and
        # its key/value convolutions are dynamic: static taps plus a per-token projection.
        for which in ("q", "k", "v"):
            sd[f"{which}_input_layernorm.weight"] = t(p + f"{which}_input_norm.weight")
        sd["self_attn.k_proj.sconv.static_weight"] = t(p + "attn.k_sconv.static_w")
        sd["self_attn.v_proj.sconv.static_weight"] = t(p + "attn.v_sconv.static_w")
        sd["self_attn.k_proj.sconv.weight_proj.weight"] = t(p + "attn.k_sconv.weight_proj")
        sd["self_attn.v_proj.sconv.weight_proj.weight"] = t(p + "attn.v_sconv.weight_proj")
    if idx > 0:
        q = t(f"model.llm.attn_res.attn_queries.{idx - 1}")
        sd["attention_input.query"] = q if q.dim() == 1 else q.reshape(-1, q.shape[-1])
    return sd

def build(idx):
    global P, NVFP4
    P, NVFP4 = f"model.llm.layers.{idx}.", idx < 20
    sliding = cfg.layer_types[idx] == "hybrid_sliding"
    # A layer inside the core is a *loop* layer: it takes `messages` as a third
    # positional argument and carries the four message gates.
    loop = cfg.loop_layer_index_start <= idx < cfg.loop_layer_index_end
    cls = {
        (True, False): M.Inkling2LoopSlidingDecoderLayer,
        (False, False): M.Inkling2LoopFullDecoderLayer,
        (True, True): M.Inkling2LoopSlidingLoopDecoderLayer,
        (False, True): M.Inkling2LoopFullLoopDecoderLayer,
    }[(sliding, loop)]
    layer = cls(cfg, idx).eval()
    layer.mlp.experts = LazyExperts(cfg, f"model.llm.layers.{idx}.", idx < 20)
    sd = state_dict(idx, sliding)
    if loop:
        # controllers.{attn,k,v,mlp} are the query, key, value and MLP gates.
        for gate, which in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            cp = f"model.llm.layers.{idx}.controllers.{which}."
            sd[f"{gate}_message_gate.down_proj.weight"] = t(cp + "down_weight")
            sd[f"{gate}_message_gate.forget_gate_proj.weight"] = t(cp + "forget_weight")
            sd[f"{gate}_message_gate.message_gate_proj.weight"] = t(cp + "gain_weight")
    miss, unexp = layer.load_state_dict(sd, strict=False)
    real = [m for m in miss if "experts" not in m]
    assert not real and not unexp, (idx, real, unexp)
    return layer, sliding

# ---- the whole model for one token ----------------------------------------
# Inkling2LoopTextModel.forward, written out: the pre-core layers, the core three
# times, then the post-core layers, with each later pass restarting from the state the
# core was entered with and taking messages from the pass before.
emb = raw("model.llm.embed.weight")[TOKEN].astype(np.float32)
enorm = raw("model.llm.embed_norm.weight").astype(np.float32)
rms = np.sqrt((emb * emb).mean() + cfg.rms_norm_eps)
x = torch.from_numpy((emb / rms) * enorm).view(1, 1, -1).float()

print("building 24 layers ...", flush=True)
layers = [build(i) for i in range(cfg.num_hidden_layers)]

start, end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
n_loop = end - start

# The checkpoint splits the reference's [n_loop, 4, hidden] message queries across two
# [2 * n_loop, hidden] tensors. Which of the pair is which message is an inference;
# INKLING2_MSG_ORDER picks it so all four readings can be tried.
qs = t("model.llm.middle_cycle_cross_pass_queries")
kvs = t("model.llm.middle_cycle_cross_pass_kv_queries")
msg_q = torch.zeros(n_loop, 4, cfg.hidden_size)
for l in range(n_loop):
    a, b = (0, 1) if ORDER[0] == "q" else (1, 0)
    c, d = (0, 1) if ORDER[2] == "k" else (1, 0)
    msg_q[l, 0] = qs[2 * l + a]        # query
    msg_q[l, 3] = qs[2 * l + b]        # mlp
    msg_q[l, 1] = kvs[2 * l + c]       # key
    msg_q[l, 2] = kvs[2 * l + d]       # value
message_input = M.Inkling2LoopMessageAttentionResidual(cfg).eval()
with torch.no_grad():
    message_input.query.copy_(msg_q)

final_input = M.Inkling2LoopBlockAttentionResidual(cfg).eval()
with torch.no_grad():
    final_input.query.copy_(t("model.llm.attn_res.output_query"))

pos = torch.arange(1).unsqueeze(0)
blocks = x.new_zeros(1, 1, 0, x.shape[-1])

TRACE = []

def run(idx, hidden, blocks, messages=None):
    layer, sliding = layers[idx]
    rel = M.get_relative_distance(hidden, None, is_sliding=sliding)
    loop = cfg.loop_layer_index_start <= idx < cfg.loop_layer_index_end
    args = (hidden, blocks, messages) if loop else (hidden, blocks)
    with torch.no_grad():
        out = layer(*args, attention_mask=None, relative_distance=rel,
                    conv_mask=None, cache=None, position_ids=pos)
    y = out[0] if isinstance(out, tuple) else out
    TRACE.append((idx, y.detach().reshape(-1).float().numpy().copy()))
    return y

for i in range(start):
    nx = run(i, x, blocks)
    if i % cfg.n_layers_per_attn_res_block == 0:
        blocks = torch.cat([blocks, x.unsqueeze(-2)], dim=-2)
    x = nx
pre_x, pre_blocks = x, blocks
n_pre = pre_blocks.shape[-2]

for i in range(start, end):
    nx = run(i, x, blocks)
    if i % cfg.n_layers_per_attn_res_block == 0:
        blocks = torch.cat([blocks, x.unsqueeze(-2)], dim=-2)
    x = nx

for p in range(1, cfg.loop_num_passes):
    prev_blocks = blocks[..., n_pre + 1:, :]
    prev_x = x
    x, blocks = pre_x, pre_blocks
    for j, i in enumerate(range(start, end)):
        with torch.no_grad():
            msgs = message_input(prev_blocks, prev_x, j)
        if p == 1 and j == 0:
            # The first messages any layer ever receives, for bisecting the loop.
            msgs.detach().reshape(-1).float().numpy().astype("<f4").tofile(
                OUT.replace(".bin", ".msg.bin"))
            prev_x.detach().reshape(-1).float().numpy().astype("<f4").tofile(
                OUT.replace(".bin", ".prevx.bin"))
            prev_blocks.detach().reshape(-1).float().numpy().astype("<f4").tofile(
                OUT.replace(".bin", ".prevblocks.bin"))
            mm = msgs.detach().reshape(4, -1).float().numpy()
            print("  pass1 layer0 messages absmax", [round(float(abs(v).max()), 4) for v in mm])
            print("    prev_x absmax", float(prev_x.abs().max()),
                  "prev_blocks", list(prev_blocks.shape))
        nx = run(i, x, blocks, msgs)
        if i % cfg.n_layers_per_attn_res_block == 0:
            blocks = torch.cat([blocks, x.unsqueeze(-2)], dim=-2)
        x = nx

for i in range(end, cfg.num_hidden_layers):
    nx = run(i, x, blocks)
    if i % cfg.n_layers_per_attn_res_block == 0:
        blocks = torch.cat([blocks, x.unsqueeze(-2)], dim=-2)
    x = nx

with torch.no_grad():
    x = final_input(blocks, x)[0]
    norm = M.Inkling2LoopRMSNorm(cfg.hidden_size, cfg.rms_norm_eps).eval()
    norm.weight.copy_(t("model.llm.norm.weight"))
    h = norm(x)
    unembed = t("model.llm.unembed.weight")
    logits = torch.nn.functional.linear(h.reshape(1, -1), unembed)
    if cfg.final_logit_softcapping:
        c = cfg.final_logit_softcapping
        logits = torch.tanh(logits / c) * c

lg = logits.reshape(-1).float().numpy()
lg.astype("<f4").tofile(OUT)
top = np.argsort(-lg[:cfg.unpadded_vocab_size])[:5]
print("order", ORDER, "-> argmax", int(top[0]), "logit", float(lg[top[0]]),
      "top5", [(int(i), round(float(lg[i]), 3)) for i in top])
print("hidden absmax", float(np.abs(x.numpy()).max()))
# Every layer execution in order, for bisecting against the engine.
np.stack([v for _, v in TRACE]).astype("<f4").tofile(OUT.replace(".bin", ".trace.bin"))
print("executions traced:", len(TRACE), "->", OUT.replace(".bin", ".trace.bin"))
for n, (idx, v) in enumerate(TRACE):
    if n < 3 or n > len(TRACE) - 3:
        print(f"  [{n:2d}] layer {idx:2d} absmax {abs(v).max():.4f} first2 {v[:2]}")
