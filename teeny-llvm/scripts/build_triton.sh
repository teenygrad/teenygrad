#!/bin/sh

set -e
checkEnv() {
    if [ -z "$1" ]; then
        echo "Error: $1 environment variable not set"
        exit 1
    fi
    
}

createDir() {
  if [ ! -d "$1" ]; then
    mkdir -p "$1"
  fi
}

checkEnv "BUILD_DIR"
checkEnv "MODULES_DIR"

createDir "$BUILD_DIR/triton"
createDir "$BUILD_DIR/install"

if [ -f "$BUILD_DIR/triton.txt" ]; then
    echo "triton.txt already exists, skipping build"
    exit 0
fi

echo "Building Triton - $BUILD_DIR/triton ($MODULES_DIR/triton)"

export LLVM_BUILD_DIR="$BUILD_DIR/llvm"
export LLVM_INCLUDE_DIRS="$BUILD_DIR/llvm/include"
export LLVM_LIBRARY_DIR="$BUILD_DIR/llvm/lib"
export LLVM_SYSPATH="$BUILD_DIR/llvm"

cmake -B "$BUILD_DIR/triton" -S "$MODULES_DIR/triton" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DTRITON_CODEGEN_BACKENDS="amd;nvidia" \
      -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"

ninja -C "$BUILD_DIR/triton" install

touch "$BUILD_DIR/triton.txt"
