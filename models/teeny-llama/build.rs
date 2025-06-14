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

fn main() {
    let dialect_libs = "MLIRAffineAnalysis;MLIRAffineDialect;MLIRAffineTransforms;MLIRAffineTransformOps;MLIRAffineUtils;MLIRAMDGPUDialect;MLIRAMDGPUTransforms;MLIRAMDGPUUtils;MLIRAMXDialect;MLIRAMXTransforms;MLIRArithDialect;MLIRArithValueBoundsOpInterfaceImpl;MLIRArithTransforms;MLIRArithUtils;MLIRArmNeonDialect;MLIRArmNeonTransforms;MLIRArmSMEDialect;MLIRArmSMETransforms;MLIRArmSVEDialect;MLIRArmSVETransforms;MLIRAsyncDialect;MLIRAsyncTransforms;MLIRBufferizationDialect;MLIRBufferizationPipelines;MLIRBufferizationTransformOps;MLIRBufferizationTransforms;MLIRComplexDialect;MLIRControlFlowDialect;MLIRControlFlowTransforms;MLIRDLTITransformOps;MLIRDLTIDialect;MLIREmitCDialect;MLIREmitCTransforms;MLIRFuncDialect;MLIRFuncTransforms;MLIRFuncTransformOps;MLIRGPUDialect;MLIRGPUTransforms;MLIRGPUTransformOps;MLIRGPUPipelines;MLIRGPUUtils;MLIRIndexDialect;MLIRIRDL;MLIRLinalgDialect;MLIRLinalgTransformOps;MLIRLinalgTransforms;MLIRLinalgUtils;MLIRLLVMIRTransforms;MLIRLLVMDialect;MLIRNVVMDialect;MLIRROCDLDialect;MLIRVCIXDialect;MLIRMathDialect;MLIRMathTransforms;MLIRMemRefDialect;MLIRMemRefTransformOps;MLIRMemRefTransforms;MLIRMemRefUtils;MLIRMeshDialect;MLIRMeshTransforms;MLIRMLProgramDialect;MLIRMLProgramTransforms;MLIRMPIDialect;MLIRNVGPUDialect;MLIRNVGPUUtils;MLIRNVGPUTransformOps;MLIRNVGPUTransforms;MLIROpenACCDialect;MLIROpenACCTransforms;MLIROpenMPDialect;MLIRPDLDialect;MLIRPDLInterpDialect;MLIRPolynomialDialect;MLIRPtrDialect;MLIRQuantDialect;MLIRQuantTransforms;MLIRQuantUtils;MLIRSCFDialect;MLIRSCFTransformOps;MLIRSCFTransforms;MLIRSCFUtils;MLIRShapeDialect;MLIRShapeOpsTransforms;MLIRSparseTensorDialect;MLIRSparseTensorPipelines;MLIRSparseTensorTransformOps;MLIRSparseTensorTransforms;MLIRSparseTensorUtils;MLIRSPIRVDialect;MLIRSPIRVModuleCombiner;MLIRSPIRVConversion;MLIRSPIRVTransforms;MLIRSPIRVUtils;MLIRTensorDialect;MLIRTensorInferTypeOpInterfaceImpl;MLIRTensorTilingInterfaceImpl;MLIRTensorTransforms;MLIRTensorTransformOps;MLIRTensorUtils;MLIRTosaDialect;MLIRTosaShardingInterfaceImpl;MLIRTosaTransforms;MLIRTransformDebugExtension;MLIRTransformDialect;MLIRTransformDialectIRDLExtension;MLIRTransformLoopExtension;MLIRTransformPDLExtension;MLIRTransformDialectTransforms;MLIRTransformDialectUtils;MLIRUBDialect;MLIRVectorDialect;MLIRVectorTransforms;MLIRVectorTransformOps;MLIRVectorUtils;MLIRX86VectorDialect;MLIRX86VectorTransforms;MLIRXeGPUDialect;MLIRXeGPUTransforms;MLIRSPIRVTarget;MLIRNVVMTarget;MLIRROCDLTarget;MLIRTestDynDialect;MLIRTosaTestPasses";
    let conversion_libs = "MLIRAffineToStandard;MLIRAMDGPUToROCDL;MLIRArithAttrToLLVMConversion;MLIRArithToAMDGPU;MLIRArithToArmSME;MLIRArithToEmitC;MLIRArithToLLVM;MLIRArithToSPIRV;MLIRArmNeon2dToIntr;MLIRArmSMEToSCF;MLIRArmSMEToLLVM;MLIRAsyncToLLVM;MLIRBufferizationToMemRef;MLIRComplexDivisionConversion;MLIRComplexToLibm;MLIRComplexToLLVM;MLIRComplexToSPIRV;MLIRComplexToStandard;MLIRControlFlowToLLVM;MLIRControlFlowToSCF;MLIRControlFlowToSPIRV;MLIRConvertToLLVMInterface;MLIRConvertToLLVMPass;MLIRFuncToEmitC;MLIRFuncToLLVM;MLIRFuncToSPIRV;MLIRGPUToGPURuntimeTransforms;MLIRGPUToLLVMSPV;MLIRGPUToNVVMTransforms;MLIRGPUToROCDLTransforms;MLIRGPUToSPIRV;MLIRIndexToLLVM;MLIRIndexToSPIRV;MLIRLinalgToStandard;MLIRLLVMCommonConversion;MLIRMathToEmitC;MLIRMathToFuncs;MLIRMathToLibm;MLIRMathToLLVM;MLIRMathToROCDL;MLIRMathToSPIRV;MLIRMemRefToEmitC;MLIRMemRefToLLVM;MLIRMemRefToSPIRV;MLIRMeshToMPI;MLIRMPIToLLVM;MLIRNVGPUToNVVM;MLIRNVVMToLLVM;MLIROpenACCToSCF;MLIROpenMPToLLVM;MLIRPDLToPDLInterp;MLIRReconcileUnrealizedCasts;MLIRSCFToControlFlow;MLIRSCFToEmitC;MLIRSCFToGPU;MLIRSCFToOpenMP;MLIRSCFToSPIRV;MLIRShapeToStandard;MLIRSPIRVAttrToLLVMConversion;MLIRSPIRVToLLVM;MLIRTensorToLinalg;MLIRTensorToSPIRV;MLIRTosaToArith;MLIRTosaToLinalg;MLIRTosaToMLProgram;MLIRTosaToSCF;MLIRTosaToTensor;MLIRUBToLLVM;MLIRUBToSPIRV;MLIRVectorToArmSME;MLIRVectorToGPU;MLIRVectorToLLVM;MLIRVectorToLLVMPass;MLIRVectorToSCF;MLIRVectorToSPIRV;MLIRVectorToXeGPU";

    let link_libs = vec![
        "triton",
        "MLIRIR",
        "LLVMCore",
        "LLVMSupport",
        "LLVMDemangle",
        "MLIRTransforms",
        "MLIRTransformUtils",
        "MLIRPass",
        "MLIROptLib",
        "MLIRSupport",
        "MLIRNVVMDialect",
        "MLIRNVVMToLLVMIRTranslation",
        "MLIRGPUToNVVMTransforms",
        "MLIRGPUToGPURuntimeTransforms",
        "MLIRGPUTransforms",
        "MLIRIR",
        "MLIRControlFlowToLLVM",
        "MLIRBytecodeWriter",
        "MLIRPass",
        "MLIRTransforms",
        "MLIRLLVMDialect",
        "MLIRSupport",
        "MLIRTargetLLVMIRExport",
        "MLIRMathToLLVM",
        "MLIRROCDLToLLVMIRTranslation",
        "MLIRGPUDialect",
        "MLIRSCFToControlFlow",
        "MLIRIndexToLLVM",
        "MLIRGPUToROCDLTransforms",
        "MLIRUBToLLVM",
        "LLVMPasses",
        "LLVMNVPTXCodeGen",
        "LLVMAMDGPUCodeGen",
        "LLVMAMDGPUAsmParser",
        "LLVMAMDGPUCodeGen",
        "LLVMAMDGPUAsmParser",
        "MLIRInferIntRangeCommon",
        "MLIRInferIntRangeInterface",
        "MLIRDataLayoutInterfaces",
        "LLVMCodeGen",
        "stdc++",
        "m",
    ];

    for lib in link_libs {
        println!("cargo:rustc-link-arg=-l{}", lib);
    }
}
