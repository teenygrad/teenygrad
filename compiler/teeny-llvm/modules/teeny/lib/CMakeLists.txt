cmake_minimum_required(VERSION 3.28)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

include_directories(${PROJECT_SOURCE_DIR})

#
# teeny
#

add_library(teeny STATIC 
  teeny.h teeny.cpp 
  compiler.h compiler.cpp
)

target_link_libraries(teeny 
  PUBLIC 
  
  ${dialect_libs}
  ${conversion_libs}  

  # triton
  triton 
  TritonTestDialectTritonGPU

  # MLIR core
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRPass
  MLIRTransforms
)

mlir_check_all_link_libraries(teeny)

# Export all dependencies
set_target_properties(teeny PROPERTIES
  EXPORT_NAME teeny
  EXPORT_PROPERTIES "LINK_LIBRARIES"
)

#
# teeny-opt
#

add_llvm_executable(teeny-opt 
  compiler.cpp compiler.h teeny-opt.cpp 
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
install(TARGETS teeny teeny-opt
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
