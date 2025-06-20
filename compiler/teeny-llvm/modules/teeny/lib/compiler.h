/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef TEENY_COMPILER_H
#define TEENY_COMPILER_H

#include <string>
#include <vector>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/ParseUtilities.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/Module.h"

namespace mlir {
  class MLIRContext;
  class ModuleOp;
}

namespace teeny {

typedef struct {
  std::string target;
  int capability;
  int numWarps;
  int threadsPerWarp;
  int numCtas;
  int numConsumerGroups;
  int numBuffersWarpSpec;
  int regDecProducer;
  int regIncConsumer;
  int numStages;
  int ptxVersion;
} NvidiaGpuConfig;

class Compiler {
  public:
    Compiler();
    ~Compiler();

    bool initLlvm();
    
    bool compile(
      const char *source, // the source code to compile (utf-8 encoded)
      const char *config, // the compiler configuration (utf-8 encoded)
      const char **target, // the target code (binary)
      int *target_size // the size of the target code (in bytes)
    );

    mlir::DialectRegistry &getRegistry() {
      return registry;
    }

    mlir::MLIRContext &getContext() {
      return context;
    }
    
  private:
    bool initialized;
    mlir::DialectRegistry registry;
    mlir::MLIRContext context;        
        
    void initContext();

    bool makeTtir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> *module);

    bool makeTtgir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
      const NvidiaGpuConfig &config, bool dumpEnabled);

    bool makeLlir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
      const NvidiaGpuConfig &config, bool enableLineInfo);

    bool makePtx(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
      const NvidiaGpuConfig &config, bool enableFpFusion);

    bool makeCubin(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module);

    void dumpModule(mlir::OwningOpRef<mlir::ModuleOp> &module);

    std::string translateLLVMIRToASM(mlir::OwningOpRef<mlir::ModuleOp> &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enableFpFusion, bool isObject, 
                                 bool enableIrDump, bool enableLLVMOpt,
                                 const std::string &enableLLVMOptFlags, bool enableTiming);

    std::unique_ptr<llvm::TargetMachine> createTargetMachine(llvm::Module *module, std::string proc,
                    bool enableFpFusion, bool enableLLVMOpt, const std::string &features);
 };
}

 #endif /* TEENY_COMPILER_H */
 