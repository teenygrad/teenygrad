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

use teeny_core::dtype::Dtype;
use teeny_macros::kernel;
use teeny_triton::triton as tl;

#[kernel]
pub fn tensor_add<D: Dtype>(
    x_ptr: &[D],
    y_ptr: &[D],
    output_ptr: &[D],
    n_elements: usize,
    #[constexpr] BLOCK_SIZE: usize,
) {
    let pid = tl::program_id(0);

    // Calculate the starting offset for this block
    let block_start = pid * BLOCK_SIZE;

    // Create offsets for the elements this block will process
    let offsets = block_start + tl::arange::<usize>(0, BLOCK_SIZE);

    // Create a mask to handle cases where n_elements is not divisible by BLOCK_SIZE
    let mask = offsets < n_elements;

    // Load data from global memory with masking
    let x = tl::load(x_ptr + offsets, mask);
    let y = tl::load(y_ptr + offsets, mask);

    // Perform element-wise addition
    let output = x + y;

    // Store result back to global memory
    tl::store(output_ptr + offsets, output, mask);
}
