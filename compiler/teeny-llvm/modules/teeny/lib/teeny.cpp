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

#include <stdio.h>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"

#include "teeny.h"

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
/*----------------------------------------*
 | Forward declarations                   |
 *----------------------------------------*/

int init_mlir(compiler_t* compiler);

/*----------------------------------------*
 | Public functions                       |
 *----------------------------------------*/

extern "C" int teeny_new(compiler_t* compiler) {
    *compiler = new compiler_t();
    if (!compiler) {
        return TEENY_FAILURE;
    }

    if (!init_mlir(*compiler)) {
        delete *compiler;
        *compiler = nullptr;
        return TEENY_FAILURE;
    }

    printf("Teeny compiler initialized\n");
    return TEENY_SUCCESS;
}

extern "C" int teeny_free(compiler_t* compiler) {
    if (*compiler) {
        delete *compiler;
        *compiler = nullptr;
    }

    return TEENY_SUCCESS;
}

/*----------------------------------------*
 | Private functions                      |
 *----------------------------------------*/

int init_mlir(compiler_t* compiler) {
    printf("Initializing MLIR\n");
    mlir::registerAllDialects(compiler->registry);

    printf("MLIR initialized\n");
    compiler->initialized = true;
    return TEENY_SUCCESS;
}
