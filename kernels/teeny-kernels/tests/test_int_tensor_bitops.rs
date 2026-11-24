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

//! Bitwise and shift operators on integer tensors (#170).
//!
//! Sub-byte quantised weights travel packed in f32 elements, since parameter buffers take the
//! graph's dtype, and a kernel reads them back out by bitcasting each word to `i32` and
//! shifting and masking out its bytes. `T::cast(.., true)` used to convert by value, `>>` did
//! not lower, and teeny-triton had no bitwise or shift operators for `i32` tensors at all.
//!
//!   cargo test -p teeny-kernels --no-default-features --features qemu,hardware,training \
//!       --test test_int_tensor_bitops

#![allow(non_snake_case)]

use teeny_core::dtype::{AddOffsets, Comparison, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

const BLOCK_SIZE: i32 = 64;

/// `y[i]` is byte `i % 4` (little-endian) of the f32 word `x[i / 4]`, as an unsigned value.
#[kernel]
pub fn unpack_bytes<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<f32>>,
    n_elements: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shr<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shr<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let i = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = i.lt(n_elements);
    let words = T::load(
        x_ptr.add_offsets(i >> 2),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let bits = T::cast::<f32, i32>(words, None, true);
    let byte = (bits >> ((i & 3) * 8)) & 0xFF;
    T::store(
        y_ptr.add_offsets(i),
        T::cast::<i32, f32>(byte, None, false),
        Some(m),
        &[],
        None,
        None,
    );
}

/// `y[i] = (((b & 0xFFF) << 3) ^ (b >> 20)) | 1` for `b` the bits of the f32 word `x[i]`. Every
/// operator appears, and `>>` meets negative words, where it must be arithmetic. The result's
/// magnitude stays under 2^24, so it is exact as an f32.
#[kernel]
pub fn mix_bits<T, const BLOCK_SIZE: i32>(
    x_ptr: InPtr<T::Pointer<f32>>,
    y_ptr: OutPtr<T::Pointer<f32>>,
    n_elements: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitOr<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitXor<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shr<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
{
    let pid = T::program_id(Axis::X);
    let i = T::arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE;
    let m = i.lt(n_elements);
    let words = T::load(
        x_ptr.add_offsets(i),
        Some(m),
        None,
        &[],
        None,
        None,
        None,
        false,
    );
    let bits = T::cast::<f32, i32>(words, None, true);
    let mixed = (((bits & 0xFFF) << 3) ^ (bits >> 20)) | 1;
    T::store(
        y_ptr.add_offsets(i),
        T::cast::<i32, f32>(mixed, None, false),
        Some(m),
        &[],
        None,
        None,
    );
}

/// Both kernels compile for the backend under test, with or without a device.
#[test]
fn test_the_bitwise_kernels_compile() -> anyhow::Result<()> {
    dotenv::dotenv().ok();
    let target = teeny_runtime::reference_target();
    teeny_runtime::compile_kernel(&UnpackBytes::new(BLOCK_SIZE), &target, true, false)?;
    teeny_runtime::compile_kernel(&MixBits::new(BLOCK_SIZE), &target, true, false)?;
    Ok(())
}

#[cfg(feature = "hardware")]
mod execution {
    use super::*;
    use teeny_core::device::Device;
    use teeny_core::device::buffer::Buffer;

    /// Deterministic words spread over every bit, negative ones included. Words whose
    /// exponent is all ones are moved off it, so no word is an infinity or a NaN, which a
    /// backend may canonicalise in passing.
    fn words(n: usize) -> Vec<u32> {
        let mut state = 0x9E37_79B9u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let w = state ^ (state >> 15);
                if (w >> 23) & 0xFF == 0xFF {
                    w ^ (1 << 23)
                } else {
                    w
                }
            })
            .collect()
    }

    /// Launches the `$kernel` kernel over the words `$x` (as f32) and returns its `$n_out`
    /// outputs, one launch lane per output.
    macro_rules! launch {
        ($kernel:ty, $x:expr, $n_out:expr) => {{
            let device = teeny_runtime::open()?;
            let x: Vec<f32> = $x.iter().map(|&w| f32::from_bits(w)).collect();
            let n_out: usize = $n_out;
            let mut x_buf = device.buffer::<f32>(x.len())?;
            let y_buf = device.buffer::<f32>(n_out)?;
            x_buf.to_device(&x)?;
            let kernel = <$kernel>::new(BLOCK_SIZE);
            let target = teeny_runtime::default_target(&device)?;
            let path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
            let program = teeny_runtime::load_program::<$kernel>(&path)?;
            let cfg = teeny_runtime::launch_config(n_out, &program);
            device.launch(
                &program,
                &cfg,
                (x_buf.as_device_ptr(), y_buf.as_device_ptr(), n_out as i32),
            )?;
            let mut y = vec![0.0f32; n_out];
            y_buf.to_host(&mut y)?;
            y
        }};
    }

    #[test]
    fn test_unpacking_reads_every_byte_of_every_word() -> anyhow::Result<()> {
        dotenv::dotenv().ok();
        let x = words(100);
        let got = launch!(UnpackBytes, x, x.len() * 4);
        let want: Vec<f32> = x
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .map(f32::from)
            .collect();
        assert_eq!(got, want);
        Ok(())
    }

    #[test]
    fn test_masks_shifts_and_xor_match_i32_arithmetic() -> anyhow::Result<()> {
        dotenv::dotenv().ok();
        let x = words(300);
        assert!(
            x.iter().any(|&w| (w as i32) < 0),
            "the words must include negative ones"
        );
        let got = launch!(MixBits, x, x.len());
        let want: Vec<f32> = x
            .iter()
            .map(|&w| {
                let b = w as i32;
                ((((b & 0xFFF) << 3) ^ (b >> 20)) | 1) as f32
            })
            .collect();
        assert_eq!(got, want);
        Ok(())
    }
}
