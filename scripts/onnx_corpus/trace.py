#!/usr/bin/env python3
"""Writes onnxruntime's value for every intermediate f32 tensor of a corpus model, for
test_onnx_corpus.rs to find the first node where teenygrad diverges.

    <model dir>/trace/
        index.tsv               name, file, element count (in graph order)
        t_<i>.bin               raw little-endian f32

Tensors larger than --max-elems are skipped (a vocabulary-sized logits tensor, say).

    python trace.py ~/.cache/teeny-onnx/corpus/functiongemma [--max-elems 2000000]
"""

import argparse
import json
import os

import numpy as np
import onnx
import onnxruntime as ort

DT = {"f32": np.float32, "i64": np.int64}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir")
    p.add_argument("--max-elems", type=int, default=2_000_000)
    p.add_argument("--chunk", type=int, default=400)
    a = p.parse_args()
    d = os.path.expanduser(a.dir)
    man = json.load(open(os.path.join(d, "manifest.json")))
    feeds = {
        t["name"]: np.fromfile(os.path.join(d, t["file"]), dtype=DT[t["dtype"]]).reshape(t["shape"])
        for t in man["inputs"]
    }
    path = os.path.join(d, "model.onnx")
    m = onnx.load(path)
    outputs = {o.name for o in m.graph.output}
    names = [o for n in m.graph.node for o in n.output if o and o not in outputs]

    out = os.path.join(d, "trace")
    os.makedirs(out, exist_ok=True)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    rows = []
    for start in range(0, len(names), a.chunk):
        want = names[start:start + a.chunk]
        mm = onnx.load(path)
        del mm.graph.output[:]
        for n in want:
            mm.graph.output.append(onnx.helper.make_empty_tensor_value_info(n))
        sess = ort.InferenceSession(mm.SerializeToString(), so, providers=["CPUExecutionProvider"])
        for n, v in zip(want, sess.run(None, feeds)):
            v = np.asarray(v)
            if v.dtype != np.float32 or v.size == 0 or v.size > a.max_elems:
                continue
            f = f"t_{len(rows)}.bin"
            np.ascontiguousarray(v).tofile(os.path.join(out, f))
            rows.append((n, f, v.size))
        print(f"  {min(start + a.chunk, len(names))}/{len(names)}", flush=True)
    with open(os.path.join(out, "index.tsv"), "w") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"wrote {len(rows)} tensors to {out}")


if __name__ == "__main__":
    main()
