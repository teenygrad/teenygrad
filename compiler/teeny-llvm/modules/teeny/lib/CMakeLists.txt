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

add_library(teeny STATIC 
  teeny.h teeny.cpp 
  compiler.h compiler.cpp
)

# Use whole-archive to ensure all symbols are included
target_link_options(teeny PRIVATE
  -Wl,--whole-archive
  $<TARGET_FILE:teeny>
  -Wl,--no-whole-archive
)

target_link_libraries(teeny 
  PRIVATE 
  
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

# Make the library self-contained
set_target_properties(teeny PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    # Ensure all symbols are exported
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
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
install(TARGETS teeny-opt
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS teeny
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        COMPONENT library)
