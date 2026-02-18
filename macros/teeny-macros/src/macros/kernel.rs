/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Ident, ItemFn};

pub fn kernel(_attrs: TokenStream, item: TokenStream) -> TokenStream {
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
        #vis #sig #block
    }
    .into();

    let kernel_name = &format!("{name}_kernel");
    let static_ident = Ident::new(kernel_name, input.sig.ident.span());
    let sig_str = quote!(#sig).to_string();
    let block_str = quote!(
      #vis extern "C" #sig #block
    )
    .to_string();
    let static_stream: TokenStream = quote! (
        #[allow(non_upper_case_globals)]
        pub static #static_ident: teeny_triton::TritonKernel = teeny_triton::TritonKernel {
            name: #kernel_name,
            sig: #sig_str,
            block_str: #block_str,
        };
    )
    .into();

    let mut result = function_stream;
    result.extend(static_stream);
    result
}
