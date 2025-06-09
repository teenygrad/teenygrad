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
checkEnv "TUTORIALS_DIR"

createDir "$BUILD_DIR/tutorials"

export INSTALL_DIR="$BUILD_DIR/install/tutorials"
createDir "$INSTALL_DIR"

export LLVM_BUILD_DIR="$BUILD_DIR/llvm"
export LLVM_INCLUDE_DIRS="$BUILD_DIR/llvm/include"
export LLVM_LIBRARY_DIR="$BUILD_DIR/llvm/lib"
export LLVM_SYSPATH="$BUILD_DIR/llvm"

cmake -B "$BUILD_DIR/tutorials" -S "$TUTORIALS_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$BUILD_DIR/llvm/lib/cmake/llvm" \
  -DMLIR_DIR="$BUILD_DIR/llvm/lib/cmake/mlir" \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
        
ninja -C "$BUILD_DIR/tutorials" install
echo "Tutorial build completed: $?"

touch "$BUILD_DIR/tutorials/build.done"