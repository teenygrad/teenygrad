cmake_minimum_required(VERSION 3.28)

# Add both static libstdc++ and static libgcc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

include_directories(${PROJECT_SOURCE_DIR})

#
# teeny
#

add_library(teeny SHARED 
  teeny.h teeny.cpp 
  compiler.h compiler.cpp
)

# Force static linking of all dependencies
target_link_options(teeny PRIVATE
  -Wl,-Bstatic
)

target_link_libraries(teeny 
  PRIVATE 
  
  -Wl,--whole-archive
  ${dialect_libs}
  ${conversion_libs}  

  # triton
  triton
  pybind11::module

  # MLIR core
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRTransforms

  # transitive dependencies
  LLVMAArch64AsmParser
  LLVMAArch64CodeGen
  LLVMARMAsmParser
  LLVMARMCodeGen
  LLVMARMDesc
  LLVMARMInfo
  LLVMAVRAsmParser
  LLVMAVRCodeGen
  LLVMAVRDesc
  LLVMAVRInfo
  LLVMBPFAsmParser
  LLVMBPFCodeGen
  LLVMBPFDesc
  LLVMBPFInfo
  LLVMHexagonAsmParser
  LLVMHexagonCodeGen
  LLVMHexagonDesc
  LLVMHexagonInfo
  LLVMLanaiAsmParser
  LLVMLanaiCodeGen
  LLVMLanaiDesc
  LLVMLanaiInfo
  LLVMLoongArchAsmParser
  LLVMLoongArchCodeGen
  LLVMLoongArchDesc
  LLVMLoongArchInfo
  LLVMMipsAsmParser
  LLVMMipsCodeGen
  LLVMMipsDesc
  LLVMMipsInfo
  LLVMMSP430AsmParser
  LLVMMSP430CodeGen
  LLVMMSP430Desc
  LLVMMSP430Info
  LLVMPowerPCAsmParser
  LLVMPowerPCCodeGen
  LLVMPowerPCDesc
  LLVMPowerPCInfo
  LLVMRISCVAsmParser
  LLVMRISCVCodeGen
  LLVMRISCVDesc
  LLVMRISCVInfo
  LLVMRISCVTargetMCA
  LLVMSparcAsmParser
  LLVMSparcCodeGen
  LLVMSparcDesc
  LLVMSparcInfo
  LLVMSPIRVCodeGen
  LLVMSPIRVDesc
  LLVMSPIRVInfo
  LLVMSystemZAsmParser
  LLVMSystemZCodeGen
  LLVMSystemZDesc
  LLVMSystemZInfo
  LLVMVEAsmParser
  LLVMVECodeGen
  LLVMVEDesc
  LLVMVEInfo
  LLVMWebAssemblyAsmParser
  LLVMWebAssemblyCodeGen
  LLVMWebAssemblyDesc
  LLVMWebAssemblyInfo
  LLVMX86AsmParser
  LLVMX86CodeGen
  LLVMX86Desc
  LLVMX86Info
  LLVMX86TargetMCA
  LLVMXCoreCodeGen
  LLVMXCoreDesc
  LLVMXCoreInfo
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  
  -Wl,--no-whole-archive
  -Wl,-Bdynamic

  Python3::Python
  TritonTestDialectTritonGPU
)

mlir_check_all_link_libraries(teeny)

# Make the library self-contained
set_target_properties(teeny PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON

    # Ensure all symbols are exported
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF

    # Set the shared library version
    VERSION 0.1.0
    SOVERSION 1
)

#
# teeny-opt
#

add_llvm_executable(teeny-opt 
  compiler.h teeny-opt.cpp 
  PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(teeny-opt)
target_link_libraries(teeny-opt PRIVATE teeny)

mlir_check_all_link_libraries(teeny-opt)

# Install headers
install(DIRECTORY ./
        DESTINATION include/teeny
        FILES_MATCHING PATTERN "*.h"
                      PATTERN "*.hpp"
                      PATTERN "*.inc"
                      PATTERN "*.def"
                      PATTERN "*.pdll"
                      PATTERN "*.td"
                      PATTERN "*.td.h")

# Install 
install(TARGETS teeny-opt
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS teeny
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT library)
