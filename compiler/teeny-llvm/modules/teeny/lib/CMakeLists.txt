cmake_minimum_required(VERSION 3.28)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

include_directories(${PROJECT_SOURCE_DIR})

add_llvm_executable(teeny-opt teeny-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(teeny-opt)
target_link_libraries(teeny-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}

  # tests
  TritonTestAnalysis
  TritonTestDialectTritonGPU
  TritonAMDGPUTestAnalysis

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(teeny-opt)

add_library(teeny STATIC 
  teeny.h teeny.cpp 
  compiler.h compiler.cpp
)

target_link_libraries(teeny PUBLIC 
  # MLIR core
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

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
install(TARGETS teeny teeny-opt
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)