/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_ARITH_MULTOADD_H
#define LIB_TRANSFORM_ARITH_MULTOADD_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_DECL_MULTOADD
#include "lib/Transform/Arith/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_ARITH_MULTOADD_H