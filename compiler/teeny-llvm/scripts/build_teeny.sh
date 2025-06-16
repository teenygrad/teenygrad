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

createDir "$BUILD_DIR/teeny"
createDir "$BUILD_DIR/install"

echo "Building Teeny - $BUILD_DIR/teeny ($MODULES_DIR/teeny)"

export CC=cc
export LLVM_BUILD_DIR="$BUILD_DIR/llvm"
export LLVM_INCLUDE_DIRS="$BUILD_DIR/llvm/include"
export LLVM_LIBRARY_DIR="$BUILD_DIR/llvm/lib"
export LLVM_SYSPATH="$BUILD_DIR/llvm"

export TRITON_SOURCE_DIR="$MODULES_DIR/triton"
export TRITON_BUILD_DIR="$BUILD_DIR/triton"

echo "AXM Triton build dir1:[${TRITON_BUILD_DIR}]"

cmake -B "$BUILD_DIR/teeny" -S "$MODULES_DIR/teeny" -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DTRITON_SOURCE_DIR="$TRITON_SOURCE_DIR" \
       -DTRITON_BUILD_DIR="$TRITON_BUILD_DIR" \
       -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"

ninja -C "$BUILD_DIR/teeny" install

# touch "$BUILD_DIR/teeny/build.done"