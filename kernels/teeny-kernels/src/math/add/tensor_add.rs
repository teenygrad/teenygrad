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

#![allow(non_snake_case)]

use teeny_macros::kernel;
use teeny_triton::triton1::*;

#[kernel]
pub fn tensor_add<T: Triton, D: Dtype>(
    x_ptr: &T::Pointer<D>,
    y_ptr: &T::Pointer<D>,
    output_ptr: &T::Pointer<D>,
    // n_elements: I32,
    // BLOCK_SIZE: I32, // uppercase implies constexpr
) {
    let pid = T::program_id(ProgramAxis::Axis0);

    // // Calculate the starting offset for this block
    // let block_start = pid * BLOCK_SIZE;

    // // Create offsets for the elements this block will process
    // let offsets = T::arange(I32(0), BLOCK_SIZE) + block_start;

    // // Create a mask to handle cases where n_elements is not divisible by BLOCK_SIZE
    // let mask = Some(offsets.lt(n_elements));

    // // Load data from global memory with masking
    // let x = T::load(x_ptr + &offsets, &mask);
    // let y = T::load(y_ptr + &offsets, &mask);

    // // Perform element-wise addition
    // let output = x + y;

    // // Store result back to global memory
    // T::store(output_ptr + &offsets, output, &mask);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dummy_triton_tensor_add() {
        todo!()
    }
}
