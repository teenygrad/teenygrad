/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_DIALECT_POLY_POLYTRAITS_H
#define LIB_DIALECT_POLY_POLYTRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::tutorial::poly {

template <typename ConcreteType>
class Has32BitArguments : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // OK to skip non-integer operand types
      if (!type.isIntOrIndex()) continue;

      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }

    return success();
  }
};

}

#endif  // LIB_DIALECT_POLY_POLYTRAITS_H