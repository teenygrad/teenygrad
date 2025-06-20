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
#include <stdlib.h>

#include "teeny.h"
#include "compiler.h"

using namespace teeny;

extern "C" TeenyStatus teeny_new(compiler_t *compiler) {
    printf("Teeny compiler initializing\n");
    Compiler *_compiler = new Compiler();
    if (!_compiler) {
        printf("Teeny compiler creation failed\n");
        return TeenyStatus::TeenyFailure;
    }

    printf("Teeny compiler initializing MLIR\n");
    if (!_compiler->initLlvm()) {
        delete _compiler;
        *compiler = nullptr;

        printf("Teeny MLIR initialization failed\n");
        return TeenyStatus::TeenyFailure;
    }

    *compiler = (void *)_compiler;
    printf("Teeny compiler initialized\n");
    return TeenyStatus::TeenySuccess;
}

extern "C" TeenyStatus teeny_free(compiler_t* compiler) {
    Compiler *_compiler = static_cast<Compiler *>(*compiler);
    if (_compiler) {
        delete _compiler;
        *compiler = nullptr;
    }

    return TeenyStatus::TeenySuccess;
}

extern "C" TeenyStatus teeny_compile(
  compiler_t compiler, // the compiler handle
  const char *source, // the source code to compile (utf-8 encoded)
  const char *config, // the compiler configuration (utf-8 encoded)
  const char **target, // the target code (binary)
  int *target_size // the size of the target code (in bytes)
) {
    Compiler *_compiler = static_cast<Compiler *>(compiler);
    if (!_compiler) {
        return TeenyStatus::TeenyFailure;
    }

    if (!_compiler->compile(source, config, target, target_size)) {
        return TeenyStatus::TeenyFailure;
    }

    return TeenyStatus::TeenySuccess;
}

extern "C" TeenyStatus teeny_free_target(char **target) {
    if (*target) {
        free(*target);
        *target = nullptr;
    }

    return TeenyStatus::TeenySuccess;
}

