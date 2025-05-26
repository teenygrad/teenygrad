#!/bin/sh

set -e

create_dir_if_not_exists() {
  if [ ! -d "$1" ]; then
    echo "Creating directory $1"
    mkdir -p "$1"
  fi
}

install_ninja() {
  if [ ! -f "/usr/bin/ninja" ]; then
    sudo apt-get install -y ninja-build
  fi
}

build_llvm() {
  cd $BUILD_DIR
  
  if [ ! -d "llvm-project" ]; then
      git clone https://github.com/llvm/llvm-project.git
      (cd llvm-project && git checkout a66376b0dc3b2ea8a84fda26faca287980986f78)
  fi

  create_dir_if_not_exists "local"
  create_dir_if_not_exists "llvm-build"

  cd llvm-build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS="llvm;clang;lld;mlir" \
      -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DLLVM_ENABLE_BINDINGS=ON \
      -DLLVM_ENABLE_OCAMLDOC=OFF \
      -DLLVM_ENABLE_SPHINX=OFF \
      -DLLVM_ENABLE_DOXYGEN=OFF \
      ../llvm-project/llvm 2>&1 | tee /tmp/llvm-build.log

  ninja
}

build_triton() {
  cd $BUILD_DIR

  if [ ! -d "triton" ]; then
    git clone https://github.com/triton-lang/triton
  fi

  create_dir_if_not_exists "triton-build"

  if [ ! -d "venv" ]; then
    python3 -m venv venv --prompt triton
  fi
  # . venv/bin/activate
  # (cd triton/python && pip install -e .)

  LLVM_INCLUDE_DIRS=$BUILD_DIR/llvm-build/include
  LLVM_LIBRARY_DIR=$BUILD_DIR/llvm-build/lib
  LLVM_SYSPATH=$BUILD_DIR/llvm-build
  MLIR_DIR=$BUILD_DIR/local/lib/cmake/mlir/MLIRConfig.cmake

  cd triton-build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="../local" \
      -DLLVM_DIR="../local/lib/cmake/llvm" \
      -DMLIR_DIR="../local/lib/cmake/mlir" \
      -DTRITON_ENABLE_AMDGPU=OFF \
      ../triton 2>&1 | tee /tmp/triton-build.log

  ninja
}

PROJECT_DIR=$(pwd | sed 's/\/teenygrad.*//')
BUILD_DIR=$PROJECT_DIR/teenygrad-deps

create_dir_if_not_exists $BUILD_DIR

install_ninja

build_llvm
build_triton
