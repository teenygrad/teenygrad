/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
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

#include "compiler.h"

#include "triton/third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/third_party/nvidia/include/NVGPUToLLVM/Passes.h"
#include "triton/third_party/nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassRegistry.h"

#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"

#include "triton/third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/third_party/nvidia/include/NVGPUToLLVM/Passes.h"
#include "triton/third_party/nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"


Compiler::Compiler() {
   initialized = false;
}

Compiler::~Compiler() {
   // NOP
}

bool Compiler::initMlir() {
   registerDialects();

   initialized = true;
   return true;
}

bool Compiler::compile(const char *source, const char *config, const char **output, int *output_size) {
   std::string mlirModuleStr(source);
   mlir::MLIRContext context(registry);
   llvm::LLVMContext llvmContext;

   // Parse the MLIR module
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlirModuleStr),
      llvm::SMLoc());
  
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    printf("Failed to parse MLIR module");
    return false;
  }

  printf("Parsed MLIR module\n");

  if (!makeTtir(&context, &module)) {
    printf("Failed to run TTIR passes on MLIR module");
    return false;
  }

  printf("Ran passes on MLIR module\n");

  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    printf("Failed to translate MLIR to LLVM IR");
    return false;
  }

  // makeLlir(context, llvmModule, false, capability, ptxVersion);
  printf("Translated MLIR to LLVM IR\n");

  // Initialize NVPTX target
  auto targetTriple = "nvptx64-nvidia-cuda";
  llvmModule->setTargetTriple(targetTriple);

  std::string targetError;
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, targetError);
  if (!target) {
    printf("Failed to lookup NVPTX target: %s", targetError.c_str());
    return false;
  }

  printf("Lookup NVPTX target\n");

  // Set target options
  llvm::TargetOptions opt;
  auto RM = llvm::Reloc::Model::PIC_;
  auto targetMachine = target->createTargetMachine(targetTriple, "sm_50", "+ptx60", opt, RM);
  llvmModule->setDataLayout(targetMachine->createDataLayout());

  printf("Set data layout\n");

  // Generate PTX
  llvm::SmallVector<char, 0> ptxBuffer;
  llvm::raw_svector_ostream ptxStream(ptxBuffer);
  
  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(
          pass, ptxStream, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
    printf("Failed to generate PTX");
    return false;
  }

  pass.run(*llvmModule);

  printf("Added passes to emit file\n");

  // Return the PTX as a memory buffer
  auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
      llvm::StringRef(ptxBuffer.data(), ptxBuffer.size()));
  *output = buffer->getBuffer().data();
  *output_size = buffer->getBuffer().size();

  printf("PTX output size: %d\n", buffer->getBuffer().size());

  return true;
}

mlir::DialectRegistry &Compiler::getRegistry() {
   return registry;
}

void Compiler::registerDialects() {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::registerTritonNvidiaGPUPasses();

  // mlir::test::registerTestAliasPass();
  // mlir::test::registerTestAlignmentPass();
  // mlir::test::registerTestAllocationPass();
  // mlir::test::registerTestMembarPass();

  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::registerLLVMDIScope();

  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);  

  mlir::printRegisteredPasses();

  registry
      .insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
              mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
              mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
              mlir::NVVM::NVVMDialect, mlir::triton::nvgpu::NVGPUDialect,
              mlir::ROCDL::ROCDLDialect>();
}

bool Compiler::makeTtir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> *module) {
   mlir::PassManager pm(context);
   
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::triton::createRewriteTensorPointerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::triton::createCombineOpsPass());
  pm.addPass(mlir::triton::createReorderBroadcastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::triton::createLoopUnrollPass());

  if (mlir::failed(pm.run(**module))) {
    printf("Failed to run passes on MLIR module");
    return false;
  }

  return true;
}

bool Compiler::makeTtgir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
  const std::string &target, int capability, int numWarps, int threadsPerWarp, int numCtas, 
  int numConsumerGroups, int numBuffersWarpSpec, int regDecProducer, int regIncConsumer, 
  int numStages, bool dumpEnabled) 
{
  mlir::PassManager pm(context);

  int majorVersion = capability / 10;

  pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(target, numWarps, threadsPerWarp, numCtas));
  pm.addPass(mlir::triton::gpu::createTritonGPUCoalesce());
  if (majorVersion >= 8) {
     pm.addPass(mlir::triton::gpu::createTritonGPUF32DotTC());
  }
  
  mlir::triton::nvidia_gpu::ClusterInfo *clusterInfo = nullptr; // AXM TODO - get this from the config
  pm.addPass(mlir::createTritonNvidiaGPUPlanCTAPass(clusterInfo));
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeThreadLocality());
  pm.addPass(mlir::triton::gpu::createTritonGPUAccelerateMatmul());
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());

  mlir::triton::gpu::TritonGPUOptimizeDotOperandsOptions dotOperandsOptions;
  dotOperandsOptions.hoistLayoutConversion = capability >= 80;
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(dotOperandsOptions));

  pm.addPass(mlir::createCSEPass());

  mlir::triton::gpu::TritonGPUWSLoweringOptions wsLoweringOptions;
  wsLoweringOptions.numConsumerGroups = numConsumerGroups;

  mlir::triton::gpu::TritonGPUPipelineOptions pipelineOptions;
  pipelineOptions.numStages = numStages;
  pipelineOptions.dumpIntermediateSteps = dumpEnabled;

  mlir::triton::gpu::TritonGPUWSTaskPartitionOptions wsTaskPartitionOptions;
  wsTaskPartitionOptions.numConsumerGroups = numConsumerGroups;

  mlir::triton::gpu::TritonGPUTaskIdPropagateOptions taskIdPropagateOptions;
  taskIdPropagateOptions.numConsumerGroups = numConsumerGroups;

  mlir::triton::gpu::TritonGPUWSDataPartitionOptions wsDataPartitionOptions;
  wsDataPartitionOptions.numConsumerGroups = numConsumerGroups;

  mlir::triton::gpu::TritonGPUWSCodePartitionOptions wsCodePartitionOptions;
  wsCodePartitionOptions.numBuffers = numBuffersWarpSpec;
  wsCodePartitionOptions.numConsumerGroups = numConsumerGroups;
  wsCodePartitionOptions.regDecProducer = regDecProducer;
  wsCodePartitionOptions.regIncConsumer = regIncConsumer;

  mlir::triton::gpu::TritonGPUPingPongSyncOptions pingPongSyncOptions;
  pingPongSyncOptions.numConsumerGroups = numConsumerGroups;

  if (majorVersion == 8 || majorVersion == 9) {
    pm.addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::triton::gpu::createTritonGPUWSTaskPartition(wsTaskPartitionOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUTaskIdPropagate(taskIdPropagateOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUWSDataPartition(wsDataPartitionOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUWSCodePartition(wsCodePartitionOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUPipeline(pipelineOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUPingPongSync(pingPongSyncOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUWSLowering(wsLoweringOptions));
  } else if (majorVersion >= 10) {
    pm.addPass(mlir::triton::gpu::createTritonGPUFuseNestedLoops());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeAccumulatorInit());
    pm.addPass(mlir::triton::gpu::createTritonGPUWSTaskPartition(wsTaskPartitionOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUTaskIdPropagate(taskIdPropagateOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUWSDataPartition(wsDataPartitionOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUWSCodePartition());
    pm.addPass(mlir::triton::gpu::createTritonGPUPipeline(pipelineOptions));
    pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
    pm.addPass(mlir::createTritonNvidiaGPUPromoteLHSToTMemPass());
    pm.addPass(mlir::createTritonNvidiaGPUKeepAccInTMemPass());
    pm.addPass(mlir::triton::gpu::createTritonGPUWSLowering(wsLoweringOptions));
    pm.addPass(mlir::createCanonicalizerPass());
  } else {
    pm.addPass(mlir::createLoopInvariantCodeMotionPass());
  }

  pm.addPass(mlir::triton::gpu::createTritonGPUPrefetch());
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(dotOperandsOptions));
  pm.addPass(mlir::triton::gpu::createTritonGPUCoalesceAsyncCopy());
  pm.addPass(mlir::triton::gpu::createTritonGPURemoveLayoutConversions());
  pm.addPass(mlir::triton::gpu::createTritonGPUReduceDataDuplication());
  pm.addPass(mlir::triton::gpu::createTritonGPUReorderInstructions());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  
  if (majorVersion >= 9) {
    pm.addPass(mlir::createTritonNvidiaGPUFenceInsertionPass());
    pm.addPass(mlir::createTritonNvidiaGPUTMALoweringPass());
  }

  pm.addPass(mlir::createCanonicalizerPass());

  if (majorVersion >= 9) {
    mlir::triton::gpu::TritonGPUWSCanonicalizationOptions wsCanonicalizationOptions;
    wsCanonicalizationOptions.numConsumerGroups = numConsumerGroups;

    pm.addPass(mlir::triton::gpu::createTritonGPUWSCanonicalization(wsCanonicalizationOptions));
  }
   
  if (mlir::failed(pm.run(*module))) {
    printf("Failed to run passes on MLIR module");
    return false;
  }

  return true;
}

bool Compiler::makeLlir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
  bool enableLineInfo, int capability, int ptxVersion) 
{
  mlir::PassManager pm(context);

  pm.addPass(mlir::createTritonNvidiaGPUMMALoweringPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::triton::gpu::createAllocateSharedMemory());
  pm.addPass(mlir::createTensorMemoryAllocationPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(capability, ptxVersion));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::triton::createConvertNVGPUToLLVMPass());
  pm.addPass(mlir::triton::createConvertWarpSpecializeToLLVM());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
  if (enableLineInfo) {
    pm.addPass(mlir::createLLVMDIScopePass());
  }

  if (mlir::failed(pm.run(*module))) {
    printf("Failed to run passes on MLIR module");
    return false;
  }

  return true;
}

bool Compiler::makePtx(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module) {
   mlir::PassManager pm(context);

   // todo

   if (mlir::failed(pm.run(*module))) {
    printf("Failed to run passes on MLIR module");
    return false;
  }

  return true;
}