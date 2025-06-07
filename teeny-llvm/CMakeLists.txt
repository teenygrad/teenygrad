cmake_minimum_required(VERSION 3.28)

project(teeny-llvm)

add_subdirectory(modules/llvm-project/llvm)
# add_subdirectory(modules/triton)

add_subdirectory(examples/toy)
