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

// Arithmetic operation lang items
#[lang = "mul"]
pub trait Mul<RHS = Self> {
    type Output;
    fn mul(self, rhs: RHS) -> Self::Output;
}

impl Mul for i32 {
    type Output = i32;
    fn mul(self, _rhs: i32) -> i32 {
        loop {}
    }
}

#[lang = "add"]
pub trait Add<RHS = Self> {
    type Output;
    fn add(self, rhs: RHS) -> Self::Output;
}

impl Add for i32 {
    type Output = i32;
    fn add(self, _rhs: i32) -> i32 {
        loop {}
    }
}

pub trait Dtype: 'static {}

pub enum Mask<T: Tensor<i32>> {
    None,
    Some(T),
}   

impl Dtype for i32 {}

pub trait Tensor<T: Dtype> : Add<i32>{    
}    

pub struct TensorImpl<T: Dtype> {
    x: T,
}

pub struct Pointer<T: Dtype> {
    x: T,
}

impl <D: Dtype> Add<Pointer<D>> for Pointer<D> {
    type Output = Pointer<D>;
    fn add(self, rhs: Pointer<D>) -> Pointer<D> {
        loop {}
    }
}

mod tl {
    pub use super::*;

    pub enum ProgramAxis {
        Axis0,
        Axis1,
        Axis2,
    }

    pub fn program_id(_axis: ProgramAxis) -> i32 {
        loop {}
    }

    pub fn num_programs(_axis: ProgramAxis) -> i32 {
        loop {}
    }

    pub fn load<D: Dtype, MT: Tensor<i32>>(_ptr: Pointer<D>, _mask: &Mask<MT>) -> Pointer<D> {
       loop {}
    }

    pub fn store<D: Dtype, MT: Tensor<i32>>(_ptr: Pointer<D>, _ptr1: Pointer<D>, _mask: &Mask<MT>) -> Pointer<D> {
        loop {}
    }

    pub fn arange<T: Tensor<i32>>(_start: i32, _end: i32) -> T {
        loop {}
    }

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
