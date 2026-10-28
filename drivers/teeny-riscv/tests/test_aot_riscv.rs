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

//! AOT-compiles a traced graph (Input -> SiLU) through `aot::compile_graph` into a fresh cache
//! directory and checks every compiled kernel is a RISC-V ELF shared library in that directory.

use dotenv::dotenv;
use teeny_core::{
    graph::{DtypeRepr, SymTensor},
    model::LoweringMode,
    nn::{Layer, activation::sigmoid::Silu},
};
use teeny_kernels::graph::TritonLowering;
use teeny_riscv::compiler::aot::compile_graph;
use teeny_riscv::compiler::options::Options;

#[test]
fn aot_compiles_graph_into_cache_dir() -> anyhow::Result<()> {
    dotenv().ok();

    let (input, graph) = SymTensor::input(DtypeRepr::F32, vec![None, Some(128)]);
    let _output = Layer::call(&Silu::<f32, SymTensor, 2>::default(), input);
    let graph = graph.borrow();

    let cache_dir =
        std::env::temp_dir().join(format!("teeny-riscv-aot-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache_dir);
    std::fs::create_dir_all(&cache_dir)?;

    let options = Options::parse("capability=generic-rvv1.0")?;
    let kernels = compile_graph(
        &graph,
        &TritonLowering::new(),
        LoweringMode::Inference,
        &options,
        cache_dir.to_str().expect("utf-8 temp dir"),
        false,
    )?;

    assert!(!kernels.is_empty(), "expected at least one compiled kernel");
    for kernel in &kernels {
        let path = std::path::Path::new(&kernel.library_path);
        assert!(
            path.starts_with(&cache_dir),
            "{} not written into the AOT cache dir",
            kernel.library_path
        );
        let bytes = std::fs::read(path)?;
        assert_eq!(
            &bytes[..4],
            b"\x7fELF",
            "{} is not an ELF file",
            kernel.name
        );
        // e_machine at offset 18 (u16 LE): EM_RISCV (243).
        assert_eq!(u16::from_le_bytes([bytes[18], bytes[19]]), 243);
        // The assembly is emitted too, so a runtime compile requesting it finds the cache warm.
        assert!(path.with_extension("s").is_file());
    }

    let _ = std::fs::remove_dir_all(&cache_dir);
    Ok(())
}
