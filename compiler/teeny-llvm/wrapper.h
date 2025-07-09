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

#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/Conversion.h>
#include <mlir-c/Debug.h>
#include <mlir-c/Diagnostics.h>
#include <mlir-c/Dialect/AMDGPU.h>
#include <mlir-c/Dialect/Arith.h>
#include <mlir-c/Dialect/Async.h>
#include <mlir-c/Dialect/ControlFlow.h>
#include <mlir-c/Dialect/EmitC.h>
#include <mlir-c/Dialect/Func.h>
#include <mlir-c/Dialect/GPU.h>
#include <mlir-c/Dialect/IRDL.h>
#include <mlir-c/Dialect/LLVM.h>
#include <mlir-c/Dialect/Linalg.h>
#include <mlir-c/Dialect/MLProgram.h>
#include <mlir-c/Dialect/Math.h>
#include <mlir-c/Dialect/MemRef.h>
#include <mlir-c/Dialect/NVGPU.h>
#include <mlir-c/Dialect/NVVM.h>
#include <mlir-c/Dialect/OpenMP.h>
#include <mlir-c/Dialect/PDL.h>
#include <mlir-c/Dialect/Quant.h>
#include <mlir-c/Dialect/ROCDL.h>
#include <mlir-c/Dialect/SCF.h>
#include <mlir-c/Dialect/SPIRV.h>
#include <mlir-c/Dialect/Shape.h>
#include <mlir-c/Dialect/SparseTensor.h>
#include <mlir-c/Dialect/Tensor.h>
#include <mlir-c/Dialect/Transform.h>
#include <mlir-c/Dialect/Transform/Interpreter.h>
#include <mlir-c/Dialect/Vector.h>
#include <mlir-c/ExecutionEngine.h>
#include <mlir-c/IR.h>
#include <mlir-c/IntegerSet.h>
#include <mlir-c/Interfaces.h>
#include <mlir-c/Pass.h>
#include <mlir-c/RegisterEverything.h>
#include <mlir-c/Rewrite.h>
#include <mlir-c/Support.h>
#include <mlir-c/Target/LLVMIR.h>
#include <mlir-c/Transforms.h>
 
#include "teeny/teeny.h"
