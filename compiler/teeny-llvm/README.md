# Teeny-LLVM

> ⚠️ **Warning**: This project is still under active development and not yet ready for production use.

Teeny-LLVM is a component of the [Teenygrad](https://www.teenygrad.org) project that provides a Rust interface to LLVM and Triton, with custom dialect support for additional GPU architectures supported by Teenygrad.

## Overview

This crate provides safe Rust bindings to LLVM and Triton, enabling Teenygrad to generate optimized code for various GPU architectures. It includes:

- LLVM bindings for CPU code generation
- Triton bindings for GPU code generation
- Custom dialect support for additional GPU architectures
- Build system integration with CMake and Ninja

## Requirements

- Rust toolchain
- CMake
- Ninja build system

## Building
