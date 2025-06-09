/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_DIALECT_NOISY_NOISYOPS_H
#define LIB_DIALECT_NOISY_NOISYOPS_H

#include "lib/Dialect/Noisy/NoisyDialect.h"
#include "lib/Dialect/Noisy/NoisyTypes.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#define GET_OP_CLASSES
#include "lib/Dialect/Noisy/NoisyOps.h.inc"

#endif // LIB_DIALECT_NOISY_NOISYOPS_H