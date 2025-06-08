cmake_minimum_required(VERSION 3.28)

project(mlir_tutorial)

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

set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(".")
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${PROJECT_SOURCE_DIR}/third_party)
include_directories(${PROJECT_BINARY_DIR}/third_party) # Tablegen'd files

add_subdirectory(mlir)
