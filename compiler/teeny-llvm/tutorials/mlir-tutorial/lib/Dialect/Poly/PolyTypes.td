/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_DIALECT_POLY_POLYTYPES_TD
#define LIB_DIALECT_POLY_POLYTYPES_TD

include "PolyDialect.td"
include "mlir/IR/AttrTypeBase.td"

// A base class for all types in this dialect
class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Polynomial : Poly_Type<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coefficients";

  let description = [{
    A type for polynomials with integer coefficients in a single-variable polynomial ring.
  }];

  let parameters = (ins "int":$degreeBound);
  let assemblyFormat = "`<` $degreeBound `>`";
}

#endif  // LIB_DIALECT_POLY_POLYTYPES_TD