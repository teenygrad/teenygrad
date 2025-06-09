/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_TYPES_POLY_POLYTYPES_H
#define LIB_TYPES_POLY_POLYTYPES_H

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include "mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/Poly/PolyTypes.h.inc"

#endif  // LIB_TYPES_POLY_POLYTYPES_H