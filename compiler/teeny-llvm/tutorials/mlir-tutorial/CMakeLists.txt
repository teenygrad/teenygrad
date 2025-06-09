cmake_minimum_required(VERSION 3.28)

project(mlir_tutorial)

set(CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/mlir-tutorial" CACHE PATH "Installation Directory" FORCE)
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

message(STATUS "AXM Using MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")
message(STATUS "AXM Using LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message(STATUS "AXM Using PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "AXM Using PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_BINARY_DIR}) # Tablegen'd files

# Install headers
install(DIRECTORY lib/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
                      PATTERN "*.hpp"
                      PATTERN "*.inc"
                      PATTERN "*.def"
                      PATTERN "*.pdll"
                      PATTERN "*.td"
                      PATTERN "*.td.h")

add_subdirectory(lib)
