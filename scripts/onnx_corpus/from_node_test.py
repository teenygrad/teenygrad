#!/usr/bin/env python3
"""Turns ONNX backend conformance tests into corpus cases, with traces, for debugging.

Each test directory (model.onnx, test_data_set_0/{input,output}_<i>.pb) is written as a
corpus case that test_onnx_corpus.rs runs, and trace.py's per-tensor reference is added
so ONNX_CORPUS_TRACE reports the first node that diverges:

    python from_node_test.py --out /tmp/conf-cases test_softmax_axis_1_expanded [...]

Test names are looked up in the vendored suite unless given as paths.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import onnx
from onnx import numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.join(HERE, "../../support/teeny-onnx/onnx/onnx/backend/test/data/node")
KIND = {np.dtype(np.float32): "f32", np.dtype(np.int64): "i64", np.dtype(np.int32): "i32",
        np.dtype(np.bool_): "bool"}


def convert(test, out):
    src = test if os.path.isdir(test) else os.path.join(SUITE, test)
    name = os.path.basename(os.path.normpath(src))
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    shutil.copy(os.path.join(src, "model.onnx"), os.path.join(d, "model.onnx"))
    man = {"inputs": [], "outputs": []}
    for kind in ("input", "output"):
        files = sorted(glob.glob(os.path.join(src, "test_data_set_0", f"{kind}_*.pb")),
                       key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
        for i, f in enumerate(files):
            t = onnx.TensorProto()
            t.ParseFromString(open(f, "rb").read())
            a = np.ascontiguousarray(numpy_helper.to_array(t))
            fn = f"{kind[:3] if kind == 'output' else 'in'}_{i}.bin"
            a.tofile(os.path.join(d, fn))
            man[kind + "s"].append({"name": t.name, "dtype": KIND.get(a.dtype, "f32"),
                                     "shape": list(a.shape), "file": fn})
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    subprocess.run([sys.executable, os.path.join(HERE, "trace.py"), d], check=True,
                   stdout=subprocess.DEVNULL)
    print(d)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("tests", nargs="+")
    a = p.parse_args()
    for t in a.tests:
        convert(t, os.path.expanduser(a.out))


if __name__ == "__main__":
    main()
