/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */

#ifndef MLIR_TUTORIAL_NOISY_NOISY_DIALECT_H
#define MLIR_TUTORIAL_NOISY_NOISY_DIALECT_H

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include "mlir/IR/DialectImplementation.h"

#include "lib/Dialect/Noisy/NoisyDialect.h.inc"

constexpr int INITIAL_NOISE = 12;
constexpr int MAX_NOISE = 26;

#endif // MLIR_TUTORIAL_NOISY_NOISY_DIALECT_H