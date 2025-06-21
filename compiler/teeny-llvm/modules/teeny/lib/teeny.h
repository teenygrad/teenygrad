/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
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

#ifndef TEENY_H
#define TEENY_H

#define TEENY_SUCCESS 1
#define TEENY_FAILURE 0

#ifdef __cplusplus
extern "C" {
#endif

typedef void *compiler_t;

typedef enum {
  TeenySuccess = 1,
  TeenyFailure = 0
} TeenyStatus;

//
// Create a new compiler, the argument
// will be set to the new compiler handle
//
// Returns:
//   TEENY_SUCCESS on success,
//   TEENY_FAILURE on failure
//
TeenyStatus teeny_new(compiler_t *compiler);

//
// Free the compiler instance, if successfull
// the compiler handle will be set to NULL
//
// Returns:
//   TEENY_SUCCESS on success,
//   TEENY_FAILURE on failure
//
TeenyStatus teeny_free(compiler_t *compiler);

//
// Compile the given source code into a module
//
// Returns:
//   TEENY_SUCCESS on success,
//   TEENY_FAILURE on failure
//
TeenyStatus teeny_compile(
  compiler_t compiler, // the compiler handle
  const char *source, // the source code to compile (utf-8 encoded)
  const char *config, // the compiler configuration (utf-8 encoded)
  const char **target, // the target code (binary)
  int *target_size // the size of the target code (in bytes)
);

//
// Free the target code returned by a previous call to teeny_compile, the argument
// will be set to NULL.
//
// Returns:
//   TEENY_SUCCESS on success,
//   TEENY_FAILURE on failure
//
TeenyStatus teeny_free_target(char **target);

#ifdef __cplusplus
}
#endif

#endif // TEENY_H