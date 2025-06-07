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

use teeny_macros::kernel;

#[kernel]
fn my_kernel(x: i32) -> i32 {
    // Invalid return type
    let _ = x * 2;
    return x;
}

#[test]
fn test_kernel_macro_snapshot() {
    let output = my_kernel(100);
    insta::assert_snapshot!(output.to_string());
}
