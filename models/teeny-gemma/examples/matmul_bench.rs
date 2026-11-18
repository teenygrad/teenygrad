/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Times FunctionGemma's linear kernel (`linear4`) alone, at the shapes one decode step
//! and one prompt chunk use, and reports milliseconds per call and weight bandwidth.
//!
//! `matmul_bench [iterations] [blocks] [kernels]`: `blocks` a comma list of lane counts
//! (default 128), `kernels` a comma list of `rows` (program ids along the rows, the
//! default) and `reuse` (program ids down the rows, reusing weights from cache). On a K3,
//! run it once as is (X100 cores) and once after `echo $$ > /proc/set_ai_thread` (A100
//! cores). `TEENY_RISCV_THREADS` sets the threads.

use std::time::Instant;

use teeny_core::graph::{CustomData, DtypeRepr, Graph, SymTensor};
use teeny_core::model::LoweringMode;
use teeny_core::name_scope::name_scope;
use teeny_gemma::kernels::linear::Linear4Op;
use teeny_kernels::graph::TritonLowering;
use teeny_test::model::{RuntimeModel, runtime};

/// `(name, M, K, N)`: the step's rows, input features and output features.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    ("q_proj      ", 1, 640, 1024),
    ("gate/up     ", 1, 640, 2048),
    ("down_proj   ", 1, 2048, 640),
    ("lm_head     ", 1, 640, 262144),
    ("q_proj x16  ", 16, 640, 1024),
    ("gate/up x16 ", 16, 640, 2048),
    ("down x16    ", 16, 2048, 640),
    ("gate/up x64 ", 64, 640, 2048),
    ("down x64    ", 64, 2048, 640),
];

fn main() -> anyhow::Result<()> {
    let iters: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(20);
    let blocks: Vec<i32> = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "128".into())
        .split(',')
        .map(|b| b.parse())
        .collect::<Result<_, _>>()?;
    let kernels: Vec<String> = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "rows".into())
        .split(',')
        .map(str::to_string)
        .collect();
    let device = runtime::open()?;
    let target = runtime::default_target(&device)?;
    let cpus = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Cpus_allowed_list"))
                .map(|l| l.split_whitespace().last().unwrap_or("?").to_string())
        })
        .unwrap_or_else(|| "?".into());
    println!("cpus {cpus}, {iters} iterations per shape");
    println!("kernel shape        block   M     K       N   ms/call   GFLOP/s   weight GB/s");

    for kernel in &kernels {
        for &block in &blocks {
            for &(name, m, k, n) in SHAPES {
                let op = match kernel.as_str() {
                    "rows" => Linear4Op::<f32>::new(k, n, block),
                    "reuse" => Linear4Op::<f32>::new(k, n, block).with_weight_reuse(),
                    other => anyhow::bail!("unknown kernel '{other}' (rows, reuse)"),
                };
                let (x, graph) = SymTensor::input(DtypeRepr::F32, vec![Some(m), Some(k)]);
                let y = {
                    let _g = name_scope("linear");
                    x.record_custom(CustomData::new(op), &[], None)
                };
                let graph: &Graph = &graph.borrow();
                let lowering = TritonLowering::new();
                let (_, graph_to_dag) =
                    lowering.lower_with_mapping(graph, LoweringMode::Inference)?;
                let mut model = RuntimeModel::compile(
                    &device,
                    graph,
                    &lowering,
                    LoweringMode::Inference,
                    &target,
                    m,
                    false,
                )?;
                let w: Vec<f32> = (0..n * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
                let slots: Vec<(String, usize, usize)> = model.param_info_named().collect();
                anyhow::ensure!(
                    slots.len() == 1,
                    "{name}: expected one weight slot, got {slots:?}"
                );
                for (_, node, idx) in &slots {
                    model.load_param_f32(*node, *idx, &w)?;
                }
                let input: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
                let out = [graph_to_dag[y.node_id]];

                // Check the first and last outputs against the host, then warm up.
                let got = model.run(&device, &[&input], &out)?;
                for (row, col) in [(0, 0), (m - 1, n - 1)] {
                    let want: f32 = (0..k).map(|j| input[row * k + j] * w[col * k + j]).sum();
                    let y = got[0][row * n + col];
                    anyhow::ensure!(
                        (y - want).abs() <= 1e-3 * want.abs().max(1.0),
                        "{name}: y[{row}, {col}] = {y}, host says {want}"
                    );
                }

                let t = Instant::now();
                for _ in 0..iters {
                    model.run(&device, &[&input], &out)?;
                }
                let s = t.elapsed().as_secs_f64() / iters as f64;
                println!(
                    "{kernel:<6} {name} {block:>5} {m:>3} {k:>5} {n:>7} {:>9.3} {:>9.2} {:>13.2}",
                    s * 1e3,
                    2.0 * (m * k * n) as f64 / s / 1e9,
                    (k * n * 4) as f64 / s / 1e9
                );
            }
        }
    }
    Ok(())
}
