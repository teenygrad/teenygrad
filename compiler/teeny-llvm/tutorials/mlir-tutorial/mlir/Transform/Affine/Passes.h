/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_AFFINE_PASSES_H
#define LIB_TRANSFORM_AFFINE_PASSES_H

#include "lib/Transform/Affine/AffineFullUnroll.h"
#include "lib/Transform/Affine/AffineFullUnrollPatternRewrite.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Affine/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_AFFINE_PASSES_H