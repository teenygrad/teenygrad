/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TRANSFORM_NOISY_REDUCENOISEOPTIMIZER_H
#define LIB_TRANSFORM_NOISY_REDUCENOISEOPTIMIZER_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace tutorial {
namespace noisy {

#define GEN_PASS_DECL_REDUCENOISEOPTIMIZER
#include "lib/Transform/Noisy/Passes.h.inc"

}  // namespace noisy
}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_TRANSFORM_NOISY_REDUCENOISEOPTIMIZER_H