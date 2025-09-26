/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::fs::File;
use std::io::Write;

use rustc_driver::{TimePassesCallbacks, run_compiler};

use crate::{compiler::target::Target, error::Result};

#[derive(Debug, Clone, Default)]
pub struct LlvmCompiler {}

impl LlvmCompiler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compile(
        &self,
        kernel: &teeny_triton::triton::TritonKernel,
        _target: &Target,
    ) -> Result<()> {
        let filename = "/tmp/kernel.txt".to_string();
        let mut file = File::create(&filename)?;

        file.write_all(LlvmCompiler::triton().as_bytes())?;
        file.write_all(kernel.block_str.as_bytes())?;

        let mut callbacks = TimePassesCallbacks::default();
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string(); // AXM FIXME: remove this once API changes
        let output = "-o /tmp/kernel.ll".to_string();
        let target = "-tnvptx64-nvidia-cuda".to_string();
        let crate_type = "--crate-type=lib".to_string();
        println!("target: {}", target);
        run_compiler(
            &[exe_name, filename, output, target, crate_type],
            &mut callbacks,
        );
        Ok(())
    }

    fn triton() -> &'static str {
        r#"
#![allow(non_camel_case_types)]
#![allow(internal_features)]
#![feature(no_core)]
#![feature(lang_items)]
#![no_core]
#![no_implicit_prelude]

#[lang = "meta_sized"]
pub trait MetaSized {}

#[lang = "pointee_sized"]
pub trait PointeeSized {}

// Required language items for no_core
#[lang = "sized"]
pub trait Sized : MetaSized + PointeeSized {}

#[lang = "copy"]
pub trait Copy {}

// Basic types
pub struct bool;
pub struct char;
pub struct i8;
pub struct i16;
pub struct i32;
pub struct i64;
pub struct i128;
pub struct isize;
pub struct u8;
pub struct u16;
pub struct u32;
pub struct u64;
pub struct u128;
pub struct usize;
pub struct f32;
pub struct f64;

// Basic traits
impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for i64 {}
impl Copy for i128 {}
impl Copy for isize {}
impl Copy for u8 {}
impl Copy for u16 {}
impl Copy for u32 {}
impl Copy for u64 {}
impl Copy for u128 {}
impl Copy for usize {}
impl Copy for f32 {}
impl Copy for f64 {}
impl Copy for bool {}
impl Copy for char {}

mod tl {
    pub enum ProgramAxis {
        Axis0,
        Axis1,
        Axis2,
    }

    pub trait Dtype: 'static {
    }

    pub trait Tensor<D: Dtype> {
    }    

    pub struct TensorImpl<D: Dtype> {
    }

    pub struct Pointer<T: Dtype> {
    }

    pub enum Mask<T: Tensor<i32>> {
        None,
        Some(T),
    }   

    pub fn program_id(_axis: ProgramAxis) -> i32 {
        0 
    }

    pub fn num_programs(_axis: ProgramAxis) -> i32 {
        0
    }

    pub fn load<D: Dtype, MT: Tensor<i32>>(ptr: Pointer<D>, _mask: &Mask<MT>) -> Pointer<D> {
       ptr
    }

    pub fn store<D: Dtype, MT: Tensor<i32>>(ptr: Pointer<D>, _ptr1: Pointer<D>, _mask: &Mask<MT>) -> Pointer<D> {
        ptr
    }

    // Note: arange function removed for no_core compatibility
    // This would need to be implemented differently in a real no_core environment

}

use crate::tl::*;
"#
    }
}

#[cfg(test)]
mod tests {
    use teeny_cuda::target::{Capability, CudaTarget};

    use super::*;

    #[test]
    fn test_compile() {
        let compiler = LlvmCompiler::new();
        let tensor_add = &teeny_kernels::math::add::tensor_add_kernel;
        let target = Target::Cuda(CudaTarget::new(Capability::Sm89));
        let result = compiler.compile(tensor_add, &target);
        assert!(result.is_ok());
    }
}
