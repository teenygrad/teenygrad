/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_ARITH_PASSES_H
#define LIB_TRANSFORM_ARITH_PASSES_H

#include "lib/Transform/Arith/MulToAdd.h"
#include "lib/Transform/Arith/MulToAddPdll.h"

namespace mlir {
namespace tutorial {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Arith/Passes.h.inc"

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_ARITH_PASSES_H