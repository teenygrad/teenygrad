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
use syn::{parse_macro_input, Ident, ItemFn};

pub fn proc_kernel(_attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let name = input.sig.ident.to_string();
    let vis = input.vis;
    let sig = input.sig.clone();
    let block = input.block;
    let attrs = input.attrs;

    let function_stream: TokenStream = quote! {
        #[allow(non_snake_case)]
        #[allow(clippy::too_many_arguments)]
        #(#attrs)*
        #vis #sig {
            #block
        }
    }
    .into();

    let kernel_name = &format!("{}_kernel", name);
    let static_ident = Ident::new(kernel_name, input.sig.ident.span());
    let sig_str = quote!(#sig).to_string();
    let block_str = quote!({
      #sig {
            #block
        }
    })
    .to_string();
    let static_stream: TokenStream = quote! {
        #[allow(non_upper_case_globals)]
        pub static #static_ident: teeny_triton::triton::TritonKernel = teeny_triton::triton::TritonKernel {
            name: #kernel_name,
            sig: #sig_str,
            block_str: #block_str,
        };
    }
    .into();

    let mut result = function_stream;
    result.extend(static_stream);
    result
}
