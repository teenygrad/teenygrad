cmake_minimum_required(VERSION 3.28)

project(teeny)

set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE PATH "Installation Directory" FORCE)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_DEPS ON)

if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

message(STATUS "Using MLIR_DIR: ${MLIR_DIR}")

find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR}) # Tablegen'd files

add_subdirectory(lib)

# Function to combine multiple libraries into one
function(combine_static_libs output_library)
    set(libraries ${ARGN})
    
    # Create a dummy source file
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp "")
    
    add_library(${output_library} STATIC ${CMAKE_CURRENT_BINARY_DIR}/dummy.cpp)
    
    if(APPLE)
        # Use libtool on MacOS
        set(script_contents "libtool -static -o $<TARGET_FILE:${output_library}>")
        foreach(lib ${libraries})
            set(script_contents "${script_contents} $<TARGET_FILE:${lib}>")
        endforeach()
        add_custom_command(TARGET ${output_library} POST_BUILD
            COMMAND /bin/sh -c "${script_contents}"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS ${libraries})
    else()
        # Use ar on Linux
        set(script_contents "ar crsT $<TARGET_FILE:${output_library}>")
        foreach(lib ${libraries})
            set(script_contents "${script_contents} $<TARGET_FILE:${lib}>")
        endforeach()
        add_custom_command(TARGET ${output_library} POST_BUILD
            COMMAND /bin/sh -c "${script_contents}"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS ${libraries})
    endif()
endfunction()

# Example usage (uncomment and modify as needed):
# combine_static_libs(combined_lib lib1 lib2 lib3)