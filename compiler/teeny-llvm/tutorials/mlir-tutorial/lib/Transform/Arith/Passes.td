/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_ARITH_PASSES_TD
#define LIB_TRANSFORM_ARITH_PASSES_TD

include "mlir/Dialect/PDL/IR/PDLDialect.td"
include "mlir/Dialect/PDLInterp/IR/PDLInterpOps.td"
include "mlir/Pass/PassBase.td"

def MulToAdd : Pass<"mul-to-add"> {
  let summary = "Convert multiplications to repeated additions";
  let description = [{
    Convert multiplications to repeated additions.
  }];
}

def MulToAddPdll : Pass<"mul-to-add-pdll"> {
  let summary = "Convert multiplications to repeated additions using pdll";
  let description = [{
    Convert multiplications to repeated additions (using pdll).
  }];
  let dependentDialects = [
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
];
}

#endif  // LIB_TRANSFORM_ARITH_PASSES_TD