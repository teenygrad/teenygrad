"""Reference DSpark drafter for Inkling-2-Mini, replayed against the engine's own traces.

Run the board with INKLING2_TRACE=<dir>/t.bin, which writes per step the 48 visit
outputs, the token ids and the final block states. This script rebuilds the target
features from them and, for each generated position, drafts 7 tokens and counts how
many the target's own greedy continuation would accept. The open questions of the
spec (which visit tensor, whether the embedding is normed, the mup divisor) are
choices here, settled by which gives the higher acceptance.

usage: inkling2-dspark-ref.py <trace dir> [--visit out|in] [--embed-norm 0|1] [--mup 1]
"""
import argparse, glob, json, os, re, struct
import numpy as np, torch

DIR = "/home/petr/models/Inkling-2-Mini"
H, R, B, G, HQ, HKV, HD = 2304, 256, 8, 7, 24, 8, 128
VU, EPS, THETA, CAP = 200058, 1e-6, 1e6, 30.0
VISITS = 48


def headers(files):
    out = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (f, 8 + n + v["data_offsets"][0], 8 + n + v["data_offsets"][1], v["shape"], v["dtype"])
    return out


HDR = headers(glob.glob(f"{DIR}/model-*.safetensors"))
MTP = headers([f"{DIR}/mtp.safetensors"])


def load(table, name):
    f, a, b, shape, dt = table[name]
    buf = np.fromfile(f, dtype=np.uint8, count=b - a, offset=a)
    if dt == "BF16":
        buf = (buf.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    elif dt == "F32":
        buf = buf.view(np.float32)
    return torch.from_numpy(buf.reshape(shape).copy())


def w(name):
    return load(MTP, name)


def rms(x, weight=None):
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)
    return y if weight is None else y * weight


def rope(x, pos):
    # neox: rotate halves, pairs (j, j + 64).
    inv = THETA ** (-torch.arange(0, HD, 2, dtype=torch.float64) / HD)
    ang = pos.double()[:, None] * inv[None, :]
    cos, sin = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]
    a, b = x[..., : HD // 2], x[..., HD // 2 :]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def read_trace(d):
    """Per position: the 48 visit outputs, the token id and the final blocks."""
    steps = sorted(
        int(m.group(1)) for f in os.listdir(d) if (m := re.fullmatch(r"t\.(\d+)\.ids\.bin", f))
    )
    ids, visits, blocks = [], [], []
    for at in steps:
        tok = np.fromfile(f"{d}/t.{at}.ids.bin", dtype="<u4")
        t = len(tok)
        v = np.fromfile(f"{d}/t.{at}.bin", dtype="<f4").reshape(VISITS, t, H)
        bl = np.fromfile(f"{d}/t.{at}.blocks.bin", dtype="<f4")
        nb = bl.size // (t * H) - 1
        stack = bl[: nb * t * H].reshape(t, nb, H)
        last = bl[nb * t * H :].reshape(t, H)
        ids += tok.tolist()
        visits.append(torch.from_numpy(v.transpose(1, 0, 2).copy()))
        blocks.append((torch.from_numpy(stack.copy()), torch.from_numpy(last.copy())))
    return ids, torch.cat(visits), blocks


def final_hidden(blocks):
    """The target's final-norm hidden: one block-attention residual, then the norm."""
    q = load(HDR, "model.llm.attn_res.output_query")
    nw = load(HDR, "model.llm.norm.weight")
    out = []
    for stack, last in blocks:
        s = rms(torch.cat([stack, last[:, None, :]], 1))
        a = torch.softmax((s @ q) * H**-0.5, -1)
        out.append(rms((a[..., None] * s).sum(1), nw))
    return torch.cat(out)


class Draft:
    def __init__(self, embed_norm, mup):
        self.mup = mup
        self.alpha = torch.softmax(w("fuse_weight"), -1)
        self.embed = load(HDR, "model.llm.embed.weight")
        self.embed_norm = load(HDR, "model.llm.embed_norm.weight") if embed_norm else None
        self.lm = load(HDR, "model.llm.unembed.weight")[:VU]
        self.mask = w("mask_embed")
        self.norm = w("norm.weight")
        self.L = []
        for i in range(3):
            p = f"layers.{i}."
            self.L.append({k: w(p + n) for k, n in {
                "dn": "dflare_norm.weight", "in": "input_layernorm.weight",
                "post": "post_attention_layernorm.weight",
                "q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight",
                "v": "self_attn.v_proj.weight", "o": "self_attn.o_proj.weight",
                "qn": "self_attn.q_norm.weight", "kn": "self_attn.k_norm.weight",
                "kc": "self_attn.k_proj_ctx.weight", "vc": "self_attn.v_proj_ctx.weight",
                "kcn": "self_attn.k_norm_ctx.weight",
                "g": "mlp.gate_proj.weight", "u": "mlp.up_proj.weight", "d": "mlp.down_proj.weight",
            }.items()})
        m = "markov_head."
        self.mdn = w(m + "dflare_norm.weight")
        self.w1 = w(m + "markov_w1.weight")
        self.rnn_w, self.rnn_b = w(m + "rnn_proj.weight"), w(m + "rnn_proj.bias")
        ho_w, ho_b = w(m + "hidden_out.weight"), w(m + "hidden_out.bias")
        self.fold_w = ((self.lm.double() @ ho_w.double()) / mup).float()
        self.fold_b = ((self.lm.double() @ ho_b.double()) / mup).float()

    def mix(self, feats, row, weight):
        return rms((feats * self.alpha[row][None, :, None]).sum(1), weight)

    def context_kv(self, feats, pos):
        """Each layer's context keys and values for target rows `feats` [N, 4, H]."""
        out = []
        for i, l in enumerate(self.L):
            c = self.mix(feats, i, l["dn"])
            k = rms((c @ l["kc"].T).view(-1, HKV, HD), l["kcn"])
            out.append((rope(k, pos), (c @ l["vc"].T).view(-1, HKV, HD)))
        return out

    def propose(self, anchor, P, ctx, f_feats):
        e = self.embed[anchor]
        if self.embed_norm is not None:
            e = rms(e, self.embed_norm)
        h = torch.cat([e[None], self.mask[None].expand(G, H)])
        pos = torch.arange(P, P + B)
        res = None
        for i, l in enumerate(self.L):
            if res is None:
                res, x = h, rms(h, l["in"])
            else:
                res = h + res
                x = rms(res, l["in"])
            q = rope(rms((x @ l["q"].T).view(B, HQ, HD), l["qn"]), pos)
            k = rope(rms((x @ l["k"].T).view(B, HKV, HD), l["kn"]), pos)
            v = (x @ l["v"].T).view(B, HKV, HD)
            K, V = torch.cat([ctx[i][0], k]), torch.cat([ctx[i][1], v])
            K, V = K.repeat_interleave(3, 1), V.repeat_interleave(3, 1)
            s = torch.einsum("qhd,khd->hqk", q, K) * HD**-0.5
            o = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), V).reshape(B, HQ * HD)
            res = (o @ l["o"].T) + res
            x = rms(res, l["post"])
            h = (torch.nn.functional.silu(x @ l["g"].T) * (x @ l["u"].T)) @ l["d"].T
        hidden = rms(h + res, self.norm)
        f = self.mix(f_feats[None], 3, self.mdn)[0]
        s, prev, out = torch.zeros(R), anchor, []
        for j in range(1, B):
            hj = hidden[j]
            p = self.rnn_w @ torch.cat([s, self.w1[prev], hj, f]) + self.rnn_b
            g = torch.sigmoid(p[:R])
            s = g * s + (1 - g) * torch.tanh(p[R : 2 * R])
            lat = torch.tanh(p[2 * R :])
            logits = self.lm @ (hj / self.mup) + self.fold_w @ lat + self.fold_b
            prev = int(torch.argmax(CAP * torch.tanh(logits / CAP)))
            out.append(prev)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="+")
    ap.add_argument("--visit", default="out", choices=["out", "in"])
    ap.add_argument("--embed-norm", type=int, default=1)
    ap.add_argument("--mup", type=float, default=1.0)
    ap.add_argument("--prompt-len", type=int, nargs="*")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    d = Draft(a.embed_norm, a.mup)
    total_acc, total_steps = 0, 0
    for n, path in enumerate(a.trace):
        ids, visits, blocks = read_trace(path)
        off = 0 if a.visit == "out" else -1
        feats = torch.stack([visits[:, 5 + off], visits[:, 23 + off], visits[:, 41 + off],
                             final_hidden(blocks)], 1)
        # The prompt is every step before the first one-token step.
        t_of = [len(np.fromfile(f, dtype="<u4")) for f in sorted(
            glob.glob(f"{path}/t.*.ids.bin"), key=lambda f: int(f.split(".")[-3]))]
        prompt = sum(t for t in t_of if t > 1) if t_of[0] > 1 else 1
        # `ids` holds what was run: the last generated token never was.
        N = len(ids)
        ctx_all = d.context_kv(feats, torch.arange(N))
        accs = []
        for P in range(prompt, N - G):
            ctx = [(k[:P], v[:P]) for k, v in ctx_all]
            draft = d.propose(ids[P], P, ctx, feats[P - 1])
            truth = ids[P + 1 : P + 1 + G]
            acc = 0
            while acc < G and draft[acc] == truth[acc]:
                acc += 1
            accs.append(acc)
        print(f"{path}: prompt {prompt}, {len(accs)} anchors, accepted/step "
              f"{np.mean(accs):.2f}, histogram {np.bincount(accs, minlength=G + 1).tolist()}")
        total_acc += sum(accs)
        total_steps += len(accs)
    print(f"mean accepted {total_acc / total_steps:.2f} -> {1 + total_acc / total_steps:.2f} tokens per verify")


if __name__ == "__main__":
    main()
