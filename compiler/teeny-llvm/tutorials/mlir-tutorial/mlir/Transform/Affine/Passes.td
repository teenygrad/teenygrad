/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD
#define LIB_TRANSFORM_AFFINE_PASSES_TD

include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite engine.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

#endif  // LIB_TRANSFORM_AFFINE_PASSES_TD