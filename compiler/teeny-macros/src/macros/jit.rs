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
use quote::quote;
use syn::{parse_macro_input, Item};

pub fn jit(_attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as Item);

    match input {
        Item::Fn(input_fn) => {
            // Handle individual function
            let vis = input_fn.vis;
            let sig = input_fn.sig.clone();
            let block = input_fn.block;
            let attrs = input_fn.attrs;

            let result: TokenStream = quote! {
                #[allow(non_snake_case)]
                #[allow(clippy::too_many_arguments)]
                #(#attrs)*
                #vis #sig {
                    #block
                }
            }
            .into();

            result
        }
        Item::Impl(input_impl) => {
            // Handle entire impl block - just re-quote it as-is
            let result: TokenStream = quote! {
                #input_impl
            }
            .into();

            result
        }
        _ => {
            // For any other item type, return an error
            let error = quote! {
                compile_error!("The jit macro can only be applied to functions or impl blocks");
            };
            error.into()
        }
    }
}
