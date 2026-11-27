"""Ground truth for one Inkling-2 layer, from transformers' own modules driven with the
real checkpoint weights. Experts are dequantised only for the routed slots."""
import json, struct, sys, glob
import numpy as np, torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

DIR = "/home/petr/models/Inkling-2-Mini"
LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# One token, or a comma-separated sequence: the whole prompt in one forward pass,
# which is the ground truth for what the model should predict after it.
TOKENS = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "5000").split(",")]
TOKEN = TOKENS[0]
OUT = sys.argv[3] if len(sys.argv) > 3 else "/tmp/ref_full.bin"
# Which row of each pair is which message. Two independent choices, written as the
# queries pair then the kv pair: "qm" or "mq", then "kv" or "vk". Indexing a flat
# "qkvm" by position silently reads the wrong character and swaps key with value.
ORDER = __import__("os").environ.get("INKLING2_MSG_ORDER", "qm-kv")

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
        # Group the positions by expert, as Inkling2LoopExperts does, so each expert is
        # dequantised once per call and freed. Caching them instead is ~4 GB a layer
        # once a prompt routes to most of the 128, which runs the machine out of memory.
        out = torch.zeros_like(hidden_states, dtype=torch.float32)
        hits = {}
        for pos in range(top_k_index.shape[0]):
            for slot in range(top_k_index.shape[1]):
                hits.setdefault(int(top_k_index[pos, slot]), []).append((pos, slot))
        for e, items in hits.items():
            w13 = torch.from_numpy(expert_rows(self.prefix, self.nvfp4, "w13", e))
            w2 = torch.from_numpy(expert_rows(self.prefix, self.nvfp4, "w2", e))
            rows = [p for p, _ in items]
            gu = torch.nn.functional.linear(hidden_states[rows].float(), w13)
            # w13_weight interleaves gate and up down its rows; transformers converts
            # it with Interleave(dim=1) rather than renaming it, so chunking into
            # halves pairs every gate with the wrong up.
            g, u = gu[..., 0::2], gu[..., 1::2]
            y = torch.nn.functional.linear(self.act_fn(g, u), w2)
            for r, (pos, slot) in enumerate(items):
                out[pos] += y[r] * float(top_k_weights[pos, slot])
            del w13, w2
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

# ---- transformers' own TextModel, driven by its own forward ----------------
# The hand-written loop in inkling2-ref-full.py is my transcription of
# TextModel.forward, so a misreading of the architecture is invisible: both it and
# the port would be wrong the same way. This builds the real module on a meta device,
# which costs nothing, swaps in the lazy experts, and materialises only the ~5 GB of
# non-expert weights, so transformers' own forward drives the loop.
import torch.nn as nn

with torch.device("meta"):
    model = M.Inkling2LoopTextModel(cfg)

for idx, layer in enumerate(model.layers):
    layer.mlp.experts = LazyExperts(cfg, f"model.llm.layers.{idx}.", idx < 20)

model.to_empty(device="cpu")
model.eval()

sd = {}
for idx in range(cfg.num_hidden_layers):
    sliding = cfg.layer_types[idx] == "hybrid_sliding"
    for k, v in state_dict(idx, sliding).items():
        sd[f"layers.{idx}.{k}"] = v
    if cfg.loop_layer_index_start <= idx < cfg.loop_layer_index_end:
        for gate, which in (("q", "attn"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            cp = f"model.llm.layers.{idx}.controllers.{which}."
            sd[f"layers.{idx}.{gate}_message_gate.down_proj.weight"] = t(cp + "down_weight")
            sd[f"layers.{idx}.{gate}_message_gate.forget_gate_proj.weight"] = t(cp + "forget_weight")
            sd[f"layers.{idx}.{gate}_message_gate.message_gate_proj.weight"] = t(cp + "gain_weight")

sd["embed_tokens.weight"] = t("model.llm.embed.weight")
sd["embed_tokens.embed_norm.weight"] = t("model.llm.embed_norm.weight")
sd["norm.weight"] = t("model.llm.norm.weight")
sd["final_input.query"] = t("model.llm.attn_res.output_query")

n_loop = cfg.loop_layer_index_end - cfg.loop_layer_index_start
qs = t("model.llm.middle_cycle_cross_pass_queries")
kvs = t("model.llm.middle_cycle_cross_pass_kv_queries")
msg_q = torch.zeros(n_loop, 4, cfg.hidden_size)
qpair, kvpair = ORDER.split("-")
a, b = (0, 1) if qpair == "qm" else (1, 0)
c, d = (0, 1) if kvpair == "kv" else (1, 0)
for l in range(n_loop):
    msg_q[l, 0] = qs[2 * l + a]
    msg_q[l, 3] = qs[2 * l + b]
    msg_q[l, 1] = kvs[2 * l + c]
    msg_q[l, 2] = kvs[2 * l + d]
sd["message_input.query"] = msg_q

miss, unexp = model.load_state_dict(sd, strict=False)
real_miss = [m for m in miss if "experts" not in m]
print("missing:", real_miss[:8], "..." if len(real_miss) > 8 else "")
print("unexpected:", unexp[:8], "..." if len(unexp) > 8 else "")
assert not real_miss and not unexp, "the state dict does not cover the model"

ids = torch.tensor([TOKENS])
with torch.no_grad():
    out = model(input_ids=ids, use_cache=False)
h = out.last_hidden_state[0, -1]
unembed = t("model.llm.unembed.weight")
logits = torch.nn.functional.linear(h.view(1, -1), unembed)
if cfg.final_logit_softcapping:
    cap = cfg.final_logit_softcapping
    logits = torch.tanh(logits / cap) * cap
lg = logits.reshape(-1).float().numpy()
lg.astype("<f4").tofile(OUT)
top = np.argsort(-lg[:cfg.unpadded_vocab_size])[:8]
print("order", ORDER, "-> argmax", int(top[0]), "logit", round(float(lg[top[0]]), 3))
print("top8", [(int(i), round(float(lg[i]), 2)) for i in top])
