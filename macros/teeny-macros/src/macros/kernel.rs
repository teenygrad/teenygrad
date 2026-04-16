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
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, FnArg, GenericArgument, GenericParam, Ident, ItemFn, Pat, PatType,
    PathArguments, Type, TypeParamBound,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect()
}

/// If `ty` is `HW_IDENT::SomeName<Inner>`, return `Inner`.
fn extract_pointer_inner(ty: &Type, hw_ident: &Ident) -> Option<Type> {
    if let Type::Path(tp) = ty {
        if tp.qself.is_none() {
            let segs = &tp.path.segments;
            if segs.len() == 2 && segs[0].ident == *hw_ident {
                if let PathArguments::AngleBracketed(ab) = &segs[1].arguments {
                    if ab.args.len() == 1 {
                        if let GenericArgument::Type(inner) = &ab.args[0] {
                            return Some(inner.clone());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract the ident from a bare single-segment type, e.g. `D` → `Some(D)`.
fn simple_type_ident(ty: &Type) -> Option<Ident> {
    if let Type::Path(tp) = ty {
        if tp.qself.is_none() && tp.path.segments.len() == 1 {
            let seg = &tp.path.segments[0];
            if matches!(seg.arguments, PathArguments::None) {
                return Some(seg.ident.clone());
            }
        }
    }
    None
}

fn pat_to_str(pat: &Pat) -> String {
    quote!(#pat).to_string()
}

// ── Macro implementation ──────────────────────────────────────────────────────

pub fn kernel(_attrs: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let fn_ident = input.sig.ident.clone();
    let fn_name_str = fn_ident.to_string();
    let vis = &input.vis;
    let attrs = &input.attrs;
    let sig = &input.sig;
    let block = &input.block;

    // 1. Emit the original function unchanged.
    let function_stream: TokenStream2 = quote! {
        #[allow(non_snake_case)]
        #[allow(clippy::too_many_arguments)]
        #(#attrs)*
        #vis #sig #block
    };

    // 2. Find the hardware type param — the one with a `Triton` bound.
    let hw_ident: Ident = input
        .sig
        .generics
        .params
        .iter()
        .find_map(|p| {
            if let GenericParam::Type(tp) = p {
                let is_hw = tp.bounds.iter().any(|b| {
                    if let TypeParamBound::Trait(tb) = b {
                        tb.path
                            .segments
                            .last()
                            .map(|s| s.ident == "Triton")
                            .unwrap_or(false)
                    } else {
                        false
                    }
                });
                if is_hw { Some(tp.ident.clone()) } else { None }
            } else {
                None
            }
        })
        .expect("#[kernel] requires a type parameter with a `Triton` bound");

    // 3. Collect non-hw generic params for the struct definition/usage.
    let struct_gen_params: Vec<&GenericParam> = input
        .sig
        .generics
        .params
        .iter()
        .filter(|p| !matches!(p, GenericParam::Type(tp) if tp.ident == hw_ident))
        .collect();

    let struct_gen_args: Vec<TokenStream2> = struct_gen_params
        .iter()
        .map(|p| match p {
            GenericParam::Type(tp) => {
                let i = &tp.ident;
                quote!(#i)
            }
            GenericParam::Const(cp) => {
                let i = &cp.ident;
                quote!(#i)
            }
            GenericParam::Lifetime(lp) => {
                let l = &lp.lifetime;
                quote!(#l)
            }
        })
        .collect();

    // Use angle-bracket wrappers only when there actually are generic params.
    let (struct_generics_def, struct_generics_use) = if struct_gen_params.is_empty() {
        (quote! {}, quote! {})
    } else {
        (
            quote! { < #(#struct_gen_params),* > },
            quote! { < #(#struct_gen_args),* > },
        )
    };

    // 4. Build (type-param ident → runtime type-name variable) mapping.
    //    e.g.  D: Dtype  →  (__type_name_d, D)
    let type_param_vars: Vec<(Ident, Ident)> = input
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let GenericParam::Type(tp) = p {
                if tp.ident != hw_ident {
                    let var =
                        format_ident!("__type_name_{}", tp.ident.to_string().to_lowercase());
                    return Some((tp.ident.clone(), var));
                }
            }
            None
        })
        .collect();

    // `let __type_name_d: &str = type_name::<D>()…;`
    let type_name_decls: Vec<TokenStream2> = type_param_vars
        .iter()
        .map(|(ty_id, var)| {
            quote! {
                let #var: &str = ::std::any::type_name::<#ty_id>()
                    .rsplit("::")
                    .next()
                    .unwrap_or(::std::any::type_name::<#ty_id>());
            }
        })
        .collect();

    // 5. Parse function inputs and derive per-argument code fragments.
    let fn_inputs: Vec<&PatType> = input
        .sig
        .inputs
        .iter()
        .filter_map(|a| {
            if let FnArg::Typed(pt) = a { Some(pt) } else { None }
        })
        .collect();

    // Args<'a> tuple element types for the Kernel impl.
    let args_types: Vec<TokenStream2> = fn_inputs
        .iter()
        .map(|pt| {
            if let Some(inner) = extract_pointer_inner(&pt.ty, &hw_ident) {
                quote!(*mut #inner)
            } else {
                let ty = &pt.ty;
                quote!(#ty)
            }
        })
        .collect();

    // Entry-point parameter string expressions (evaluated at runtime in new()).
    let entry_param_exprs: Vec<TokenStream2> = fn_inputs
        .iter()
        .map(|pt| {
            let name = pat_to_str(&pt.pat);
            if let Some(inner) = extract_pointer_inner(&pt.ty, &hw_ident) {
                // Pointer arg: type name is a runtime value.
                let var_opt = simple_type_ident(&inner).and_then(|id| {
                    type_param_vars
                        .iter()
                        .find(|(i, _)| *i == id)
                        .map(|(_, v)| v)
                });
                if let Some(var) = var_opt {
                    quote! { format!("{}: *mut {}", #name, #var) }
                } else {
                    // Concrete inner type — bake into the literal.
                    let inner_str = quote!(#inner).to_string();
                    let s = format!("{name}: *mut {inner_str}");
                    quote! { ::std::string::String::from(#s) }
                }
            } else {
                // Primitive — fully static.
                let ty = &pt.ty;
                let ty_str = quote!(#ty).to_string();
                let s = format!("{name}: {ty_str}");
                quote! { ::std::string::String::from(#s) }
            }
        })
        .collect();

    // Pointer-wrapping lines, e.g. `let x_ptr = Pointer(x_ptr as *mut _);`
    let ptr_conv_exprs: Vec<TokenStream2> = fn_inputs
        .iter()
        .filter(|pt| extract_pointer_inner(&pt.ty, &hw_ident).is_some())
        .map(|pt| {
            let name = pat_to_str(&pt.pat);
            let line = format!("let {name} = Pointer({name} as *mut _);");
            quote! { ::std::string::String::from(#line) }
        })
        .collect();

    // Call arguments string (just the names, joined).
    let call_args_str: String = fn_inputs
        .iter()
        .map(|pt| pat_to_str(&pt.pat))
        .collect::<Vec<_>>()
        .join(", ");

    // Call type-arg expressions, one per original generic (hw → "LlvmTriton").
    let call_type_arg_exprs: Vec<TokenStream2> = input
        .sig
        .generics
        .params
        .iter()
        .map(|p| match p {
            GenericParam::Type(tp) => {
                if tp.ident == hw_ident {
                    quote! { ::std::string::String::from("LlvmTriton") }
                } else {
                    let var = type_param_vars
                        .iter()
                        .find(|(i, _)| *i == tp.ident)
                        .map(|(_, v)| v)
                        .expect("every non-hw type param must have a type_name var");
                    quote! { ::std::string::String::from(#var) }
                }
            }
            GenericParam::Const(cp) => {
                let id = &cp.ident;
                quote! { (#id).to_string() }
            }
            GenericParam::Lifetime(_) => quote! { ::std::string::String::new() },
        })
        .collect();

    // Original function source stored verbatim.
    let original_source_str = quote!(#vis #sig #block).to_string();

    // Struct ident (PascalCase of the function name).
    let struct_ident = Ident::new(&to_pascal_case(&fn_name_str), fn_ident.span());

    // 6. Generate the struct, its constructor, and the Kernel impl.
    // PhantomData to satisfy the "type parameter never used" requirement.
    let phantom_type_params: Vec<&Ident> = type_param_vars.iter().map(|(i, _)| i).collect();
    let phantom_field = if phantom_type_params.is_empty() {
        quote! {}
    } else {
        quote! {
            _phantom: ::std::marker::PhantomData<( #(#phantom_type_params,)* )>,
        }
    };
    let phantom_init = if phantom_type_params.is_empty() {
        quote! {}
    } else {
        quote! { _phantom: ::std::marker::PhantomData, }
    };

    let struct_stream: TokenStream2 = quote! {
        pub struct #struct_ident #struct_generics_def {
            pub name: &'static str,
            /// The original kernel function source.
            pub kernel_source: ::std::string::String,
            /// The generated entry-point wrapper source.
            pub entry_point: ::std::string::String,
            /// Combined source (`kernel_source + "\n\n" + entry_point`); used by the `Kernel` trait.
            pub source: ::std::string::String,
            #phantom_field
        }

        impl #struct_generics_def #struct_ident #struct_generics_use {
            pub fn new() -> Self {
                // Declare runtime type-name variables for each type generic.
                #(#type_name_decls)*

                let __original_source: &str = #original_source_str;

                let __entry_params_str = {
                    let __parts: ::std::vec::Vec<::std::string::String> =
                        vec![ #(#entry_param_exprs),* ];
                    __parts.join(", ")
                };

                let __ptr_conv_str = {
                    let __lines: ::std::vec::Vec<::std::string::String> =
                        vec![ #(#ptr_conv_exprs),* ];
                    __lines.join("\n    ")
                };

                let __call_type_args_str = {
                    let __type_args: ::std::vec::Vec<::std::string::String> =
                        vec![ #(#call_type_arg_exprs),* ];
                    __type_args.join(", ")
                };

                let __fn_call = format!(
                    "{}::<{}>({});",
                    #fn_name_str,
                    __call_type_args_str,
                    #call_args_str,
                );

                let __body = if __ptr_conv_str.is_empty() {
                    format!("    {}", __fn_call)
                } else {
                    format!("    {}\n    {}", __ptr_conv_str, __fn_call)
                };

                let __entry_point = format!(
                    concat!(
                        "use triton::llvm::triton::num::*;\n",
                        "use triton::llvm::triton::pointer::Pointer;\n",
                        "type LlvmTriton = triton::llvm::triton::LlvmTriton;\n",
                        "\n",
                        "#[no_mangle]\n",
                        "pub extern \"C\" fn entry_point({params}) {{\n",
                        "{body}\n",
                        "}}"
                    ),
                    params = __entry_params_str,
                    body = __body,
                );

                let __kernel_source = ::std::string::String::from(__original_source);
                let __source = format!("{}\n\n{}", __kernel_source, __entry_point);
                Self {
                    name: #fn_name_str,
                    kernel_source: __kernel_source,
                    entry_point: __entry_point,
                    source: __source,
                    #phantom_init
                }
            }
        }

        impl #struct_generics_def teeny_core::context::program::Kernel
            for #struct_ident #struct_generics_use
        {
            type Args<'__a> = ( #(#args_types,)* );

            fn name(&self) -> &str {
                self.name
            }

            fn source(&self) -> &str {
                &self.source
            }

            fn kernel_source(&self) -> &str {
                &self.kernel_source
            }

            fn entry_point(&self) -> &str {
                &self.entry_point
            }
        }
    };

    let mut result: TokenStream = TokenStream::from(function_stream);
    result.extend(TokenStream::from(struct_stream));
    result
}
