/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use teeny_llvm::{teeny_free, teeny_new};

fn main() {
    println!("Hello, world!");

    unsafe {
        let mut compiler = std::ptr::null_mut();
        let status = teeny_new(&mut compiler);
        if status == teeny_llvm::TEENY_SUCCESS {
            println!("teeny_new succeeded");
        } else {
            panic!("teeny_new failed");
        }

        let status = teeny_free(&mut compiler);
        if status == teeny_llvm::TEENY_SUCCESS {
            println!("teeny_free succeeded");
        } else {
            panic!("teeny_free failed");
        }
    }
}
