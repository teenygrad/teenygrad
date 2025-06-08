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

createDir "$BUILD_DIR/llvm"
createDir "$BUILD_DIR/install"

echo "Building LLVM - $BUILD_DIR/llvm ($MODULES_DIR/llvm-project/llvm)"

cmake -B "$BUILD_DIR/llvm" -S "$MODULES_DIR/llvm-project/llvm" -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_PROJECTS='mlir' \
       -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install"

ninja -C "$BUILD_DIR/llvm" install