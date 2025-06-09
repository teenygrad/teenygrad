/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */

#ifndef LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD
#define LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD

include "mlir/Pass/PassBase.td"

def PolyToStandard : Pass<"poly-to-standard"> {
  let summary = "Lower `poly` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `poly` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tutorial::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
    "mlir::scf::SCFDialect",
  ];
}

#endif  // LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD