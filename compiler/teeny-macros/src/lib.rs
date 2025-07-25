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

use proc_macro::TokenStream;

mod macros;

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    macros::kernel::kernel(attr, item)
}

#[proc_macro_attribute]
pub fn jit(attr: TokenStream, item: TokenStream) -> TokenStream {
    macros::jit::jit(attr, item)
}

#[proc_macro_derive(JitModule, attributes(tensor, module))]
pub fn jit_module_derive(input: TokenStream) -> TokenStream {
    macros::jit_module::jit_module_derive(input)
}

#[proc_macro]
pub fn expr(input: TokenStream) -> TokenStream {
    macros::expr::expr(input)
}
