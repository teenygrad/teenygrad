/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#include "teeny.h"
#include <stdio.h>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"

struct compiler_t {
    // Core compiler state
    bool initialized;
    
    // MLIR context and registry
    mlir::DialectRegistry registry;
    
    // Compiler options
    struct {
        bool debug;
        bool optimize;
        int optimization_level;
    } options;
    
    // Error handling
    struct {
        bool has_error;
        char* error_message;
    } error;
};

void init_mlir(compiler_t* compiler) {
    printf("Initializing MLIR\n");
    mlir::registerAllDialects(compiler->registry);

    printf("MLIR initialized\n");
    compiler->initialized = true;
}

extern "C" compiler_t* teeny_compiler_new(void) {
    compiler_t* compiler = new compiler_t();
    
    init_mlir(compiler);

    printf("Teeny compiler initialized\n");
    return compiler;
}

extern "C" void teeny_compiler_free(compiler_t* compiler) {
    delete compiler;
}