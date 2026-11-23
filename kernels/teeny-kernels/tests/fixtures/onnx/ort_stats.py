#!/usr/bin/env python3
"""Per-tensor statistics for every intermediate of an ONNX model, from onnxruntime.

Summaries rather than values: enough to find the first node whose output differs from
another runtime's, without holding gigabytes. Writes TSV of
name, count, min, max, mean, mean_abs.
"""
import sys
import numpy as np
import onnx
import onnxruntime as ort

model_path, left_p, right_p, out_p = sys.argv[1:5]
chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 800

m = onnx.load(model_path, load_external_data=False)
graph_outputs = {o.name for o in m.graph.output}
names = []
for n in m.graph.node:
    for o in n.output:
        if o and o not in graph_outputs:
            names.append(o)
print(f"{len(names)} intermediate tensors", flush=True)

left = np.fromfile(left_p, dtype=np.float32).reshape(1, 3, 96, 128)
right = np.fromfile(right_p, dtype=np.float32).reshape(1, 3, 96, 128)

so = ort.SessionOptions()
so.log_severity_level = 3
rows = []
for start in range(0, len(names), chunk):
    want = names[start : start + chunk]
    # A fresh model whose outputs are the slice of interest.
    mm = onnx.load(model_path, load_external_data=False)
    del mm.graph.output[:]
    for name in want:
        mm.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    sess = ort.InferenceSession(mm.SerializeToString(), so, providers=["CPUExecutionProvider"])
    vals = sess.run(None, {"left": left, "right": right})
    for name, v in zip(want, vals):
        a = np.asarray(v).ravel()
        if a.size == 0:
            rows.append((name, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        f = a.astype(np.float64)
        rows.append((name, a.size, float(f.min()), float(f.max()), float(f.mean()), float(np.abs(f).mean())))
    print(f"  {start + len(want)}/{len(names)}", flush=True)
    del sess, mm, vals

with open(out_p, "w") as fh:
    fh.write("name\tcount\tmin\tmax\tmean\tmean_abs\n")
    for r in rows:
        fh.write("\t".join(str(x) for x in r) + "\n")
print(f"wrote {len(rows)} rows to {out_p}")
