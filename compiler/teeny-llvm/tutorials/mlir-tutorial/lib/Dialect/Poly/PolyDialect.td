/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */

#ifndef LIB_DIALECT_POLY_POLYDIALECT_TD
#define LIB_DIALECT_POLY_POLYDIALECT_TD

include "mlir/IR/OpBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let summary = "A dialect for polynomial math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::tutorial::poly";

  let useDefaultTypePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif  // LIB_DIALECT_POLY_POLYDIALECT_TD