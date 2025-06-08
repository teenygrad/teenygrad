/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_NOISY_PASSES_H
#define LIB_TRANSFORM_NOISY_PASSES_H

#include "lib/Transform/Noisy/ReduceNoiseOptimizer.h"

namespace mlir {
namespace tutorial {
namespace noisy {

#define GEN_PASS_REGISTRATION
#include "lib/Transform/Noisy/Passes.h.inc"

}  // namespace noisy
}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_NOISY_PASSES_H