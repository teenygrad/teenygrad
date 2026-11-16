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

//! Comparison kernels must compile at every integer width, not just 32-bit (teenygrad#142).
//!
//! ONNX comparisons produce `Bool`, which lowers to a byte-wide tensor, so `Equal`/`Greater`/...
//! have to instantiate at `u8`. Each builds its result from `T::zeros_like(a)`, and teenyc used
//! to splat an `i32` zero into the narrower tensor, which MLIR's verifier rejects ("'tt.splat' op
//! requires the same element type for all operands and results"). Compile-only, so no device is
//! needed; `i32` and `f32` are the controls that always worked.

use dotenv::dotenv;
use teeny_kernels::nn::tensor::elemwise_binary::{
    ElemwiseEqualForward, ElemwiseGreaterEqualForward, ElemwiseGreaterForward,
    ElemwiseLessEqualForward, ElemwiseLessForward,
};

const BLOCK_SIZE: i32 = 1024;

macro_rules! compiles_at {
    ($test_name:ident, $kernel:ident, [$($ty:ty),+ $(,)?]) => {
        #[test]
        fn $test_name() -> anyhow::Result<()> {
            dotenv().ok();
            let target = teeny_runtime::reference_target();
            $(
                let kernel = $kernel::<$ty>::new(BLOCK_SIZE);
                teeny_runtime::compile_kernel(&kernel, &target, true, false).map_err(|e| {
                    anyhow::anyhow!(
                        "{} failed to compile at {}: {e:#}",
                        stringify!($kernel),
                        stringify!($ty)
                    )
                })?;
            )+
            Ok(())
        }
    };
}

compiles_at!(
    equal_compiles_at_every_int_width,
    ElemwiseEqualForward,
    [u8, i8, u16, i16, i32, f32]
);
compiles_at!(
    greater_compiles_at_every_int_width,
    ElemwiseGreaterForward,
    [u8, i8, u16, i16, i32, f32]
);
compiles_at!(
    greater_equal_compiles_at_every_int_width,
    ElemwiseGreaterEqualForward,
    [u8, i8, u16, i16, i32, f32]
);
compiles_at!(
    less_compiles_at_every_int_width,
    ElemwiseLessForward,
    [u8, i8, u16, i16, i32, f32]
);
compiles_at!(
    less_equal_compiles_at_every_int_width,
    ElemwiseLessEqualForward,
    [u8, i8, u16, i16, i32, f32]
);
