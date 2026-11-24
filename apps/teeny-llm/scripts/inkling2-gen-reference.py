"""Emit (input, weights, expected output) fixtures from the transformers reference
modules, so the Rust kernels are checked against first-party code rather than a
host reference that could share a misreading with the kernel."""
import json, math, os, struct, sys
import torch
from transformers.models.inkling2_loop import modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

OUT = sys.argv[1]
torch.manual_seed(20261123)

cfg_json = json.load(open("/home/petr/models/Inkling-2-Mini/config.json"))["text_config"]
# Small stand-in config: the kernels are shape-generic, and a 2304-wide fixture would
# be megabytes. Keep the real scalars (thresholds, caps, eps) and shrink the shapes.
cfg = Inkling2LoopTextConfig(**{**cfg_json,
    "hidden_size": 32, "num_hidden_layers": 4, "vocab_size": 128,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
    "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 8,
    "d_rel": 4, "rel_extent": 16, "intermediate_size": 12,
    "n_routed_experts": 8, "num_experts_per_tok": 3,
    "local_layer_ids": [0, 2], "n_layers_per_attn_res_block": 2,
    "num_hidden_layers": 6,
    "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3,
                    "gate_rank": 6, "is_varloop": False},
})

blobs, manifest = [], {}
def put(name, t):
    t = t.detach().float().contiguous().reshape(-1)
    manifest[name] = {"offset": sum(len(b) // 4 for b in blobs), "len": t.numel()}
    blobs.append(t.numpy().astype("<f4").tobytes())

def shape(name, t):
    manifest[name + ".shape"] = {"dims": list(t.shape)}

cases = {}
def case(name, inputs, params, expected, scalars=None):
    for k, v in inputs.items():
        put(f"{name}.in.{k}", v); shape(f"{name}.in.{k}", v)
    for k, v in params.items():
        put(f"{name}.w.{k}", v); shape(f"{name}.w.{k}", v)
    put(f"{name}.out", expected); shape(f"{name}.out", expected)
    cases[name] = {"scalars": scalars or {}}

H, T, NH, HD, DR = cfg.hidden_size, 6, cfg.num_attention_heads, cfg.head_dim, cfg.d_rel
x = torch.randn(1, T, H)

# --- norms -----------------------------------------------------------------
n = M.Inkling2LoopRMSNorm(H, cfg.rms_norm_eps)
with torch.no_grad(): n.weight.copy_(torch.randn(H) * 0.3)
case("rms_norm", {"x": x}, {"weight": n.weight}, n(x), {"eps": cfg.rms_norm_eps})

un = M.Inkling2LoopUnweightedRMSNorm(cfg.rms_norm_eps)
case("unweighted_rms_norm", {"x": x}, {}, un(x), {"eps": cfg.rms_norm_eps})

# --- softcap / swiglu ------------------------------------------------------
sc = M.Inkling2LoopSoftcap(7.5)
case("softcap", {"x": x * 4}, {}, sc(x * 4), {"cap": 7.5})

sw = M.Inkling2LoopSoftcappedSwiGLU(cfg)
g = torch.randn(1, T, cfg.intermediate_size * 2) * 3
gg, uu = g.chunk(2, dim=-1)
case("swiglu", {"gate": gg, "up": uu}, {}, sw(gg, uu),
     {"gate_softcap": cfg.mlp_gate_softcap, "up_softcap": cfg.mlp_up_softcap,
      "intermediate": cfg.intermediate_size})

# --- L2 clips --------------------------------------------------------------
clip = M.Inkling2LoopL2Clip(cfg.qk_l2_clip_threshold)
hv = torch.randn(1, T, NH, HD) * 5
case("l2_clip", {"x": hv}, {}, clip(hv), {"threshold": cfg.qk_l2_clip_threshold})

sclip = M.Inkling2LoopScaledL2Clip(NH, HD, cfg.qk_l2_clip_threshold)
with torch.no_grad(): sclip.scale.copy_(torch.randn(NH, HD).abs() + 0.2)
case("scaled_l2_clip", {"x": hv}, {"scale": sclip.scale}, sclip(hv),
     {"threshold": cfg.qk_l2_clip_threshold})

pos = torch.arange(T).unsqueeze(0)
lclip = M.Inkling2LoopLogScaledL2Clip(cfg, NH, DR, cfg.rel_l2_clip_threshold)
with torch.no_grad(): lclip.scale.copy_(torch.randn(NH, DR).abs() + 0.2)
rv = torch.randn(1, T, NH, DR) * 4
case("log_scaled_l2_clip", {"x": rv, "position_ids": pos.float()}, {"scale": lclip.scale},
     lclip(rv, pos), {"threshold": cfg.rel_l2_clip_threshold,
                      "alpha": cfg.log_scaling_alpha, "n_floor": cfg.log_scaling_n_floor})

# --- relative position bias ------------------------------------------------
rb = M.Inkling2LoopRelativePositionBias(cfg, NH, cfg.rel_extent, log_scaling=False)
with torch.no_grad():
    rb.linear.weight.copy_(torch.randn(NH * DR, H) * 0.1)
    rb.norm.scale.copy_(torch.randn(NH, DR).abs() + 0.2)
    rb.proj.copy_(torch.randn(DR, cfg.rel_extent) * 0.3)
rel_dist = M.get_relative_distance(x, None, is_sliding=False)
case("rel_bias", {"x": x, "relative_distance": rel_dist.float(), "position_ids": pos.float()},
     {"linear": rb.linear.weight, "scale": rb.norm.scale, "proj": rb.proj},
     rb(x, rel_dist, pos),
     {"num_heads": NH, "d_rel": DR, "rel_extent": cfg.rel_extent,
      "threshold": cfg.rel_l2_clip_threshold})

# --- short convolutions ----------------------------------------------------
conv = M.Inkling2LoopShortConvolution(H, cfg.sconv_kernel_size, layer_idx=0, conv_idx=0)
with torch.no_grad(): conv.conv1d.weight.copy_(torch.randn(H, 1, cfg.sconv_kernel_size) * 0.4)
case("sconv", {"x": x}, {"weight": conv.conv1d.weight}, conv(x),
     {"kernel": cfg.sconv_kernel_size})

NKV = cfg.num_key_value_heads
KVW = NKV * cfg.head_dim
dconv = M.Inkling2LoopDynamicShortConvolution(KVW, H, NKV, cfg.sconv_kernel_size, 0, 0)
with torch.no_grad():
    dconv.static_weight.copy_(torch.randn(cfg.sconv_kernel_size, KVW) * 0.4)
    dconv.weight_proj.weight.copy_(torch.randn(NKV * cfg.sconv_kernel_size, H) * 0.2)
kv = torch.randn(1, T, KVW)
case("dyn_sconv", {"x": kv, "dynamic_input": x},
     {"static_weight": dconv.static_weight, "weight_proj": dconv.weight_proj.weight},
     dconv(kv, x), {"kernel": cfg.sconv_kernel_size, "num_heads": NKV})

# --- router ----------------------------------------------------------------
r = M.Inkling2LoopTopKRouter(cfg)
with torch.no_grad():
    r.weight.copy_(torch.randn(cfg.n_routed_experts, H) * 0.2)
    r.e_score_correction_bias.copy_(torch.randn(cfg.n_routed_experts) * 0.5)
_, rw, rid = r(x.view(-1, H))
put("router.out.weights", rw); shape("router.out.weights", rw)
put("router.out.ids", rid.float()); shape("router.out.ids", rid)
case("router", {"x": x.view(-1, H)},
     {"weight": r.weight, "bias": r.e_score_correction_bias},
     rw, {"n_experts": cfg.n_routed_experts, "top_k": cfg.num_experts_per_tok,
          "route_scale": cfg.route_scale})

# --- block attention residual / message gate / message residual ------------
S = 2
bstates = torch.randn(1, T, S, H)
ba = M.Inkling2LoopBlockAttentionResidual(cfg, num_queries=3)
with torch.no_grad(): ba.query.copy_(torch.randn(3, H) * 0.5)
outs = ba(bstates, x)
for i, o in enumerate(outs):
    put(f"block_attn.out{i}", o); shape(f"block_attn.out{i}", o)
case("block_attn", {"states": bstates, "partial": x}, {"query": ba.query}, outs[0],
     {"states": S, "hidden": H, "queries": 3})

mg = M.Inkling2LoopMessageGate(cfg)
with torch.no_grad():
    mg.down_proj.weight.copy_(torch.randn(cfg.loop_gate_rank, 2 * H) * 0.2)
    mg.forget_gate_proj.weight.copy_(torch.randn(H, cfg.loop_gate_rank) * 0.2)
    mg.message_gate_proj.weight.copy_(torch.randn(H, cfg.loop_gate_rank) * 0.2)
msg = torch.randn(1, T, H)
case("message_gate", {"x": x, "message": msg},
     {"down": mg.down_proj.weight, "forget": mg.forget_gate_proj.weight,
      "gain": mg.message_gate_proj.weight},
     mg(x, msg), {"rank": cfg.loop_gate_rank, "eps": cfg.rms_norm_eps})

n_loop = cfg.loop_layer_index_end - cfg.loop_layer_index_start
mr = M.Inkling2LoopMessageAttentionResidual(cfg)
with torch.no_grad(): mr.query.copy_(torch.randn(n_loop, 4, H) * 0.5)
case("message_residual", {"states": bstates, "partial": x}, {"query": mr.query},
     mr(bstates, x, 1), {"states": S, "hidden": H, "loop_layer": 1, "n_loop": n_loop})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT + ".bin", "wb") as f:
    for b in blobs: f.write(b)
json.dump({"tensors": manifest, "cases": cases}, open(OUT + ".json", "w"), indent=1)
print("cases:", len(cases))
print("floats:", sum(len(b) for b in blobs) // 4, "bytes:", sum(len(b) for b in blobs))
for k in sorted(cases): print(" ", k)
