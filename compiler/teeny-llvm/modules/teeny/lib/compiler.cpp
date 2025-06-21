/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
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
#include "mlir/Target/LLVM/ModuleToObject.h"

#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/LinkAllIR.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

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
#include "mlir/Target/LLVMIR/Export.h"
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

namespace teeny {
  
Compiler::Compiler() {
   initialized = false;
}

Compiler::~Compiler() {
   // NOP
}

bool Compiler::initLlvm() {
   printf("Initializing LLVM targets...\n");
   // Initialize LLVM targets
   llvm::InitializeAllTargetInfos();
   llvm::InitializeAllTargets();
   llvm::InitializeAllTargetMCs();
   llvm::InitializeAllAsmPrinters();
   llvm::InitializeAllAsmParsers();

   // Initialize NVPTX target specifically if available
#if LLVM_HAS_NVPTX_TARGET
   printf("Initializing NVPTX target...\n");
   LLVMInitializeNVPTXTargetInfo();
   LLVMInitializeNVPTXTarget();
   LLVMInitializeNVPTXTargetMC();
   printf("NVPTX target initialization complete\n");
#else
   printf("NVPTX target not available in this build\n");
#endif

   initContext();

   initialized = true;
   printf("LLVM initialization complete\n");
   return true;
}

bool Compiler::compile(const char *source, const char *_config, const char **output, int *output_size) {
   if (!initialized) {
     printf("Compiler not initialized. Call initLlvm() first.\n");
     return false;
   }

   std::string mlirModuleStr(source);
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

  printf("Ran makeTtir on MLIR module\n");

  NvidiaGpuConfig config;

  /* Nvidia 3090 GPU*/
  config.target = "cuda:86";
  config.capability = 86;
  config.ptxVersion = 86;
  config.numWarps = 4;
  config.threadsPerWarp = 32;
  config.numCtas = 1;
  config.numConsumerGroups = 1;
  config.numBuffersWarpSpec = 1;
  config.regDecProducer = 0;
  config.regIncConsumer = 0;
  config.numStages = 3;

  if (!makeTtgir(&context, module, config, false)) {
    printf("Failed to run TTGIR passes on MLIR module");
    return false;
  }

  printf("Ran makeTtgir on MLIR module\n");

  if (!makeLlir(&context, module, config, false)) {
    printf("Failed to run LLIR passes on MLIR module");
    return false;
  }

  printf("Ran makeLlir on MLIR module\n");

  if (!makePtx(&context, module, config, false)) {
    printf("Failed to run PTX passes on MLIR module");
    return false;
  }

  printf("Finished Dumping MLIR module\n");
  dumpModule(module);

  // // Initialize NVPTX target
  // auto targetTriple = "nvptx64-nvidia-cuda";
  // module.setTargetTriple(targetTriple);

  // std::string targetError;
  // auto target = llvm::TargetRegistry::lookupTarget(targetTriple, targetError);
  // if (!target) {
  //   printf("Failed to lookup NVPTX target: %s", targetError.c_str());
  //   return false;
  // }

  // printf("Lookup NVPTX target\n");

  // // Set target options
  // llvm::TargetOptions opt;
  // auto RM = llvm::Reloc::Model::PIC_;
  // auto targetMachine = target->createTargetMachine(targetTriple, "sm_50", "+ptx60", opt, RM);
  // module->setDataLayout(targetMachine->createDataLayout());

  // printf("Set data layout\n");

  // // Generate PTX
  // llvm::SmallVector<char, 0> ptxBuffer;
  // llvm::raw_svector_ostream ptxStream(ptxBuffer);
  
  // llvm::legacy::PassManager pass;
  // if (targetMachine->addPassesToEmitFile(
  //         pass, ptxStream, nullptr, llvm::CodeGenFileType::AssemblyFile)) {
  //   printf("Failed to generate PTX");
  //   return false;
  // }

  // pass.run(*module);

  // printf("Added passes to emit file\n");

  // // Return the PTX as a memory buffer
  // auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
  //     llvm::StringRef(ptxBuffer.data(), ptxBuffer.size()));
  // *output = buffer->getBuffer().data();
  // *output_size = buffer->getBuffer().size();

  // printf("PTX output size: %d\n", buffer->getBuffer().size());

  return true;
}

void Compiler::initContext() {
  registry.insert<mlir::triton::TritonDialect, mlir::triton::gpu::TritonGPUDialect,
    mlir::math::MathDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect,
    mlir::gpu::GPUDialect, mlir::NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect, 
    mlir::LLVM::LLVMDialect, mlir::ub::UBDialect, mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
    mlir::triton::nvgpu::NVGPUDialect>();

  mlir::LLVM::registerInlinerInterface(registry);
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::LLVM::registerInlinerInterface(registry);
  
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
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
  const NvidiaGpuConfig &config, bool dumpEnabled) 
{
  mlir::PassManager pm(context);

  int majorVersion = config.capability / 10;

  pm.addPass(mlir::triton::createConvertTritonToTritonGPUPass(config.target, config.numWarps, config.threadsPerWarp, config.numCtas));
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
  dotOperandsOptions.hoistLayoutConversion = config.capability >= 80;
  pm.addPass(mlir::triton::gpu::createTritonGPUOptimizeDotOperands(dotOperandsOptions));

  pm.addPass(mlir::createCSEPass());

  mlir::triton::gpu::TritonGPUWSLoweringOptions wsLoweringOptions;
  wsLoweringOptions.numConsumerGroups = config.numConsumerGroups;

  mlir::triton::gpu::TritonGPUPipelineOptions pipelineOptions;
  pipelineOptions.numStages = config.numStages;
  pipelineOptions.dumpIntermediateSteps = dumpEnabled;

  mlir::triton::gpu::TritonGPUWSTaskPartitionOptions wsTaskPartitionOptions;
  wsTaskPartitionOptions.numConsumerGroups = config.numConsumerGroups;

  mlir::triton::gpu::TritonGPUTaskIdPropagateOptions taskIdPropagateOptions;
  taskIdPropagateOptions.numConsumerGroups = config.numConsumerGroups;

  mlir::triton::gpu::TritonGPUWSDataPartitionOptions wsDataPartitionOptions;
  wsDataPartitionOptions.numConsumerGroups = config.numConsumerGroups;

  mlir::triton::gpu::TritonGPUWSCodePartitionOptions wsCodePartitionOptions;
  wsCodePartitionOptions.numBuffers = config.numBuffersWarpSpec;
  wsCodePartitionOptions.numConsumerGroups = config.numConsumerGroups;
  wsCodePartitionOptions.regDecProducer = config.regDecProducer;
  wsCodePartitionOptions.regIncConsumer = config.regIncConsumer;

  mlir::triton::gpu::TritonGPUPingPongSyncOptions pingPongSyncOptions;
  pingPongSyncOptions.numConsumerGroups = config.numConsumerGroups;

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
    wsCanonicalizationOptions.numConsumerGroups = config.numConsumerGroups;

    pm.addPass(mlir::triton::gpu::createTritonGPUWSCanonicalization(wsCanonicalizationOptions));
  }
   
  if (mlir::failed(pm.run(*module))) {
    printf("Failed to run passes on MLIR module");
    return false;
  }

  return true;
}

bool Compiler::makeLlir(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
  const NvidiaGpuConfig &config, bool enableLineInfo) 
{
  mlir::PassManager pm(context);

  pm.addPass(mlir::createTritonNvidiaGPUMMALoweringPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUCombineTensorSelectAndIf());
  pm.addPass(mlir::triton::gpu::createTritonGPUAllocateWarpGroups());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::triton::gpu::createAllocateSharedMemory());
  pm.addPass(mlir::createTensorMemoryAllocationPass());
  pm.addPass(mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass());
  pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPass(config.capability, config.ptxVersion));
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

bool Compiler::makePtx(mlir::MLIRContext *context, mlir::OwningOpRef<mlir::ModuleOp> &module, 
  const NvidiaGpuConfig &config, bool enableFpFusion) {
  mlir::PassManager pm(context);

  std::string triple = "nvptx64-nvidia-cuda";
  std::string proc = "sm_" + std::to_string(config.capability);
  std::string features = "nvptx-short-ptr";
  printf("Starting to translate LLVM IR to PTX\n");
  std::string ret = translateLLVMIRToASM(module, triple, proc, features, {features}, 
    enableFpFusion, false, false, false, std::string(""), false);

  printf("PTX output: %s\n", ret.c_str());

  // # Find kernel names (there should only be one)
  // names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
  // assert len(names) == 1
  // metadata["name"] = names[0]
  // # post-process
  // ptx_version = f'{ptx_version//10}.{ptx_version%10}'
  // ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
  // ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
  // # Remove the debug flag that prevents ptxas from optimizing the code
  // ret = re.sub(r",\s*debug|debug,\s*", "", ret)
  // if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":
  //     print("// -----// NVPTX Dump //----- //")
  //     print(ret)

  return true;
}

void Compiler::dumpModule(mlir::OwningOpRef<mlir::ModuleOp> &module) {
  module->print(llvm::errs());
}

std::string Compiler::translateLLVMIRToASM(mlir::OwningOpRef<mlir::ModuleOp> &module,
                                 const std::string &triple,
                                 const std::string &proc,
                                 const std::string &features,
                                 const std::vector<std::string> &flags,
                                 bool enableFpFusion, bool isObject, 
                                 bool enableIrDump, bool enableLLVMOpt,
                                 const std::string &enableLLVMOptFlags, bool enableTiming) {
  using namespace mlir;

  // options
  auto options = llvm::cl::getRegisteredOptions();
  for (std::string flag : flags) {
    auto *shortPtr = static_cast<llvm::cl::opt<bool> *>(options[flag]);
    assert(shortPtr);
    shortPtr->setValue(true);
  }

  if (enableIrDump) {
    auto optIt = options.find("print-after-all");
    if (optIt != options.end()) {
      auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
      *optPtr = true;
    }
  }
  
  if (enableLLVMOpt) {
    // Check to see if we are passing a list of flags to disable optimizations.
    auto flagList = enableLLVMOptFlags;
    if (!flagList.empty()) {
      llvm::SmallVector<StringRef, 3> split;
      StringRef(flagList.c_str()).split(split, ',');
      for (auto flag : split) {
        auto optIt = options.find(flag);
        if (optIt != options.end()) {
          auto optPtr = static_cast<llvm::cl::opt<bool> *>(optIt->second);
          *optPtr = true;
        }
      }
    }
  }

  // Translate the module to LLVM IR.
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVM IR\n";
    return "";
  }

  // inline everything
  for (llvm::Function &f : llvmModule->functions())
    if (!f.hasFnAttribute(llvm::Attribute::NoInline))
      f.addFnAttr(llvm::Attribute::AlwaysInline);

  // verify and store llvm
  llvm::legacy::PassManager pm;
  pm.add(llvm::createAlwaysInlinerLegacyPass());
  pm.add(llvm::createVerifierPass());

  if (enableTiming) {
    llvm::TimePassesIsEnabled = true;
    llvm::TimePassesPerRun = true;
  }

  printf("Running passes on LLVM IR - 1\n");
  pm.run(*llvmModule);

  printf("Finished running passes on LLVM IR - 1\n");

  SmallString<0> timePassesStr;
  llvm::raw_svector_ostream reportStream(timePassesStr);

  if (enableTiming) {
    llvm::reportAndResetTimings(&reportStream);
    llvm::dbgs() << reportStream.str();
    timePassesStr.clear();
  }
  // module->print(llvm::outs(), nullptr);

  printf("Creating target machine\n");
  // create machine
  llvmModule->setTargetTriple(triple);
  auto machine = createTargetMachine(llvmModule.get(), proc, enableFpFusion, enableLLVMOpt, features);

  if (!machine) {
    printf("Failed to create target machine, cannot proceed with PTX generation\n");
    return "";
  }

  printf("Setting data layout\n");
  // set data layout
  llvmModule->setDataLayout(machine->createDataLayout());

  // emit machine code
  std::string result;

  {
    llvm::raw_string_ostream stream(result);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager pass;

    // emit
    auto fileType = isObject ? llvm::CodeGenFileType::ObjectFile
                             : llvm::CodeGenFileType::AssemblyFile;
    machine->addPassesToEmitFile(pass, pstream, nullptr, fileType);

    printf("Running passes on LLVM IR - 2\n");
    pass.run(*llvmModule);

    printf("Finished running passes on LLVM IR - 2\n");

    if (enableTiming) {
      reportAndResetTimings(&reportStream);
      llvm::dbgs() << reportStream.str();
      timePassesStr.clear();
    }
  }

  return result;
}

std::unique_ptr<llvm::TargetMachine> Compiler::createTargetMachine(llvm::Module *module, std::string proc,
                    bool enableFpFusion, bool enableLLVMOpt, const std::string &features) {
  std::string error;
  printf("Looking up target for triple: %s\n", module->getTargetTriple().c_str());
  auto target = llvm::TargetRegistry::lookupTarget(module->getTargetTriple(), error);
  if (!target) {
    printf("Failed to lookup NVPTX target: %s\n", error.c_str());
    return nullptr;
  }

  printf("Successfully found target: %s\n", target->getName());

  llvm::TargetOptions opt;
  if (enableFpFusion)
    opt.AllowFPOpFusion = llvm::FPOpFusion::Fast;

  opt.UnsafeFPMath = false;
  opt.NoInfsFPMath = false;
  opt.NoNaNsFPMath = true;
  opt.TrapUnreachable = true;
  opt.MCOptions.AsmVerbose = true;
  opt.MCOptions.PreserveAsmComments = true;

  printf("Creating target machine with proc: %s, features: %s\n", proc.c_str(), features.c_str());
  std::unique_ptr<llvm::TargetMachine> machine{target->createTargetMachine(
      module->getTargetTriple(), proc, features, opt, llvm::Reloc::PIC_,
      std::nullopt,
      enableLLVMOpt ? llvm::CodeGenOptLevel::None
                     : llvm::CodeGenOptLevel::Aggressive)};

  if (!machine) {
    printf("Failed to create target machine\n");
    return nullptr;
  }

  printf("Successfully created target machine\n");
  return machine;
}

}