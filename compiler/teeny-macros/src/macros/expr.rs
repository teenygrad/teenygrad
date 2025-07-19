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
use proc_macro2::Literal;
use quote::quote;
use syn::parse_macro_input;

pub fn expr(item: TokenStream) -> TokenStream {
    // Parse the input TokenStream into a syntax tree
    let input = parse_macro_input!(item as syn::Expr);

    // Helper function to recursively walk the expression tree and print node type

    let result = walk_expr(&input);
    result.into()
}

fn walk_expr(expr: &syn::Expr) -> proc_macro2::TokenStream {
    match expr {
        syn::Expr::Binary(bin) => {
            let left = walk_expr(&bin.left);
            let right = walk_expr(&bin.right);

            quote! { #left #bin.op #right }
        }
        // syn::Expr::Unary(un) => {
        //     let expr = walk_expr(&un.expr);
        //     quote! { #un.op #expr }
        // }
        // syn::Expr::Lit(lit) => process_lit(lit),
        // // syn::Expr::Path(_) => {
        // //     quote! { #expr }
        // // }
        // syn::Expr::Call(call) => {
        //     let func = walk_expr(&call.func);
        //     let args = call.args.iter().map(walk_expr).collect::<Vec<_>>();
        //     quote! { #func(#(#args),*) }
        // }
        // syn::Expr::Paren(paren) => {
        //     let expr = walk_expr(&paren.expr);
        //     quote! { (#expr) }
        // }
        _ => {
            quote! { #expr }
        }
    }
}

fn process_lit(lit: &syn::ExprLit) -> proc_macro2::TokenStream {
    match &lit.lit {
        syn::Lit::Int(int) => {
            quote! { teeny_core::graph::scalar(#int as f32) }
        }
        syn::Lit::Float(float) => {
            quote! { teeny_core::graph::scalar(#float as f32) }
        }
        syn::Lit::Str(str) => {
            quote! { &#str }
        }
        _ => {
            quote! { #lit }
        }
    }
}
