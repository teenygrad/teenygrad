/*
 * Copyright 2025 Jeremy Kun. All rights reserved.
 */
 
#ifndef LIB_ANALYSIS_REDUCENOISEANALYSIS_REDUCENOISEANALYSIS_H
#define LIB_ANALYSIS_REDUCENOISEANALYSIS_REDUCENOISEANALYSIS_H

#include "llvm/ADT/DenseMap.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"      // from @llvm-project

namespace mlir {
namespace tutorial {

class ReduceNoiseAnalysis {
 public:
  ReduceNoiseAnalysis(Operation *op);
  ~ReduceNoiseAnalysis() = default;

  /// Return true if a reduce_noise op should be inserted after the given
  /// operation, according to the solution to the optimization problem.
  bool shouldInsertReduceNoise(Operation *op) const {
    return solution.lookup(op);
  }

 private:
  llvm::DenseMap<Operation *, bool> solution;
};

}  // namespace tutorial
}  // namespace mlir

#endif  // LIB_ANALYSIS_REDUCENOISEANALYSIS_REDUCENOISEANALYSIS_H