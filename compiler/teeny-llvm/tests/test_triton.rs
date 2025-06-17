/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
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

use teeny_llvm::{teeny_compile, teeny_free, teeny_new};

#[cfg(feature = "cuda")]
#[test]
fn test_triton_cuda() {
    // 123456
    let mlir = r#"
#loc = loc("./vector_addition.py":10:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("./vector_addition.py":10:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("./vector_addition.py":10:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("./vector_addition.py":10:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("./vector_addition.py":10:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc2)
    %c1024_i32_0 = arith.constant 1024 : i32 loc(#loc2)
    %1 = arith.extsi %0 : i32 to i64 loc(#loc2)
    %2 = arith.extsi %c1024_i32_0 : i32 to i64 loc(#loc2)
    %3 = arith.muli %1, %2 : i64 loc(#loc2)
    %c2147483647_i64 = arith.constant 2147483647 : i64 loc(#loc2)
    %c-2147483648_i64 = arith.constant -2147483648 : i64 loc(#loc2)
    %4 = arith.cmpi sle, %3, %c2147483647_i64 : i64 loc(#loc2)
    %5 = arith.cmpi sge, %3, %c-2147483648_i64 : i64 loc(#loc2)
    %6 = arith.andi %4, %5 : i1 loc(#loc2)
    %7 = arith.muli %0, %c1024_i32_0 : i32 loc(#loc2)
    %8 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc3)
    %9 = tt.splat %7 : i32 -> tensor<1024xi32> loc(#loc4)
    %10 = arith.extsi %9 : tensor<1024xi32> to tensor<1024xi64> loc(#loc4)
    %11 = arith.extsi %8 : tensor<1024xi32> to tensor<1024xi64> loc(#loc4)
    %12 = arith.addi %10, %11 : tensor<1024xi64> loc(#loc4)
    %c2147483647_i64_1 = arith.constant 2147483647 : i64 loc(#loc4)
    %c-2147483648_i64_2 = arith.constant -2147483648 : i64 loc(#loc4)
    %cst = arith.constant dense<2147483647> : tensor<1024xi64> loc(#loc4)
    %13 = arith.cmpi sle, %12, %cst : tensor<1024xi64> loc(#loc4)
    %cst_3 = arith.constant dense<-2147483648> : tensor<1024xi64> loc(#loc4)
    %14 = arith.cmpi sge, %12, %cst_3 : tensor<1024xi64> loc(#loc4)
    %15 = arith.andi %13, %14 : tensor<1024xi1> loc(#loc4)
    %16 = arith.addi %9, %8 : tensor<1024xi32> loc(#loc4)
    %17 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc5)
    %18 = arith.cmpi slt, %16, %17 : tensor<1024xi32> loc(#loc5)
    %19 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc6)
    %20 = tt.addptr %19, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc6)
    %21 = tt.load %20, %18 : tensor<1024x!tt.ptr<f32>> loc(#loc7)
    %22 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc8)
    %23 = tt.addptr %22, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc8)
    %24 = tt.load %23, %18 : tensor<1024x!tt.ptr<f32>> loc(#loc9)
    %25 = arith.addf %21, %24 : tensor<1024xf32> loc(#loc10)
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc11)
    %27 = tt.addptr %26, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc11)
    tt.store %27, %25, %18 : tensor<1024x!tt.ptr<f32>> loc(#loc12)
    tt.return loc(#loc13)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("./vector_addition.py":19:24)
#loc2 = loc("./vector_addition.py":24:24)
#loc3 = loc("./vector_addition.py":25:41)
#loc4 = loc("./vector_addition.py":25:28)
#loc5 = loc("./vector_addition.py":27:21)
#loc6 = loc("./vector_addition.py":30:24)
#loc7 = loc("./vector_addition.py":30:16)
#loc8 = loc("./vector_addition.py":31:24)
#loc9 = loc("./vector_addition.py":31:16)
#loc10 = loc("./vector_addition.py":32:17)
#loc11 = loc("./vector_addition.py":34:26)
#loc12 = loc("./vector_addition.py":34:35)
#loc13 = loc("./vector_addition.py":34:4)
"#;

    unsafe {
        let mut compiler = std::ptr::null_mut();
        let status = teeny_new(&mut compiler);
        assert_eq!(status, teeny_llvm::TEENY_SUCCESS);

        let mut ptx_output: *mut ::std::os::raw::c_char = std::ptr::null_mut();
        let mut ptx_output_size: ::std::os::raw::c_int = 0;

        // Convert MLIR string to C string - since it's a literal, we can use CStr
        let mlir_cstr = std::ffi::CString::new(mlir).unwrap();
        let mlir = mlir_cstr.as_ptr();

        // Convert empty config string to C string
        let config_cstr = std::ffi::CString::new("").unwrap();
        let config = config_cstr.as_ptr();

        let status = teeny_compile(
            compiler,
            mlir,
            config,
            &mut ptx_output as *mut *mut i8 as *mut *const i8,
            &mut ptx_output_size,
        );

        assert_eq!(status, teeny_llvm::TEENY_SUCCESS);

        let status = teeny_free(&mut compiler);
        assert_eq!(status, teeny_llvm::TEENY_SUCCESS);
    }

    todo!("compile to nVidia and test PTX");
}
