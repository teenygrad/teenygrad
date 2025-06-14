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

 #ifndef TEENY_COMPILER_H
 #define TEENY_COMPILER_H

#include "mlir/IR/DialectRegistry.h"

class Compiler {
  public:
    Compiler();
    ~Compiler();

    bool init_mlir();
    
    bool compile(
      const char *source, // the source code to compile (utf-8 encoded)
      const char *config, // the compiler configuration (utf-8 encoded)
      char **target, // the target code (binary)
      int *target_size // the size of the target code (in bytes)
    );

    bool free_target(char **target);
    
    mlir::DialectRegistry &get_registry();
    
  private:
    bool initialized;
    mlir::DialectRegistry registry;        

    void registerTritonDialects();
 };

 #endif /* TEENY_COMPILER_H */
 