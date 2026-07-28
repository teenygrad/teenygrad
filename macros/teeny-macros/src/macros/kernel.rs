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
    Expr, FnArg, GenericArgument, GenericParam, Ident, ItemFn, MetaNameValue, Pat, PatType,
    PathArguments, Token, Type, TypeParamBound, parse::Parser, parse_macro_input,
    punctuated::Punctuated,
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
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
    {
        let segs = &tp.path.segments;
        if segs.len() == 2
            && segs[0].ident == *hw_ident
            && let PathArguments::AngleBracketed(ab) = &segs[1].arguments
            && ab.args.len() == 1
            && let GenericArgument::Type(inner) = &ab.args[0]
        {
            return Some(inner.clone());
        }
    }
    None
}

/// Extract the ident from a bare single-segment type, e.g. `D` → `Some(D)`.
fn simple_type_ident(ty: &Type) -> Option<Ident> {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && tp.path.segments.len() == 1
    {
        let seg = &tp.path.segments[0];
        if matches!(seg.arguments, PathArguments::None) {
            return Some(seg.ident.clone());
        }
    }
    None
}

fn pat_to_str(pat: &Pat) -> String {
    quote!(#pat).to_string()
}

/// The set of concrete scalar dtypes permitted by a type-parameter trait bound.
///
/// Used when a kernel opts into dispatch (via `dtypes`/`backward`) but omits an
/// explicit `dtypes` list: "no dtypes specified" means "every dtype the bound
/// allows". Only dtypes with concrete Rust impls are included (e.g. `f16`/`bf16`
/// are marker-only and cannot be monomorphized). Returns `None` for an unknown
/// bound.
fn all_dtypes_for_bound(bound: &str) -> Option<&'static [&'static str]> {
    Some(match bound {
        "Float" => &["f32", "f64"],
        "Int" => &["i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64"],
        "Num" => &[
            "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64",
        ],
        "Bool" => &["bool"],
        "Dtype" => &[
            "bool", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "f32", "f64",
        ],
        _ => return None,
    })
}

/// Map a dtype ident (e.g. `f32`) to its `DtypeRepr` variant path.
/// Returns `None` for idents that are not valid scalar dtypes.
fn dtype_ident_to_repr(id: &Ident) -> Option<TokenStream2> {
    let variant = match id.to_string().as_str() {
        "bool" => "Bool",
        "i8" => "I8",
        "i16" => "I16",
        "i32" => "I32",
        "i64" => "I64",
        "u8" => "U8",
        "u16" => "U16",
        "u32" => "U32",
        "u64" => "U64",
        "f16" => "F16",
        "bf16" => "BF16",
        "f32" => "F32",
        "f64" => "F64",
        _ => return None,
    };
    let v = format_ident!("{}", variant);
    Some(quote! { teeny_core::graph::DtypeRepr::#v })
}

/// Parsed `#[kernel(...)]` attribute arguments.
#[derive(Default)]
struct KernelAttrs {
    /// Declared supported dtypes (idents such as `f32`, `f64`, `i32`).
    dtypes: Vec<Ident>,
    /// Optional paired backward kernel struct ident.
    backward: Option<Ident>,
}

/// Parse the attribute tokens of `#[kernel(dtypes = [..], backward = Foo)]`.
fn parse_kernel_attrs(attrs: TokenStream) -> Result<KernelAttrs, syn::Error> {
    let mut out = KernelAttrs::default();
    let tokens: TokenStream2 = attrs.into();
    if tokens.is_empty() {
        return Ok(out);
    }
    let parsed = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse2(tokens)?;
    for nv in parsed {
        let key = nv
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        match key.as_str() {
            "dtypes" => {
                let Expr::Array(arr) = &nv.value else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`dtypes` must be a list, e.g. `dtypes = [f32, f64]`",
                    ));
                };
                for elem in &arr.elems {
                    let Expr::Path(p) = elem else {
                        return Err(syn::Error::new_spanned(
                            elem,
                            "each dtype must be a bare type name, e.g. `f32`",
                        ));
                    };
                    let Some(id) = p.path.get_ident() else {
                        return Err(syn::Error::new_spanned(
                            elem,
                            "each dtype must be a single identifier",
                        ));
                    };
                    if dtype_ident_to_repr(id).is_none() {
                        return Err(syn::Error::new_spanned(
                            id,
                            format!("`{id}` is not a known scalar dtype"),
                        ));
                    }
                    if out.dtypes.iter().any(|d| d == id) {
                        return Err(syn::Error::new_spanned(
                            id,
                            format!("duplicate dtype `{id}` in `dtypes`"),
                        ));
                    }
                    out.dtypes.push(id.clone());
                }
            }
            "backward" => {
                let Expr::Path(p) = &nv.value else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`backward` must be a kernel struct name",
                    ));
                };
                let Some(id) = p.path.get_ident() else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`backward` must be a single identifier",
                    ));
                };
                out.backward = Some(id.clone());
            }
            other => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    format!(
                        "unknown `#[kernel]` argument `{other}` (expected `dtypes` or `backward`)"
                    ),
                ));
            }
        }
    }
    Ok(out)
}

// ── Macro implementation ──────────────────────────────────────────────────────

pub fn kernel(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let kernel_attrs = match parse_kernel_attrs(attrs) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let input = parse_macro_input!(item as ItemFn);
    let fn_ident = input.sig.ident.clone();
    let fn_name_str = fn_ident.to_string();
    let vis = &input.vis;
    let attrs = &input.attrs;
    let sig = &input.sig;
    let block = &input.block;

    // Doc comments (`#[doc = "..."]`, from `///`/`//!`) on the annotated fn,
    // forwarded onto the generated struct(s) below -- they're the actual
    // public item downstream users and rustdoc see, so without this the
    // fn's docs never reach anything `missing_docs` checks.
    let doc_attrs: Vec<&syn::Attribute> =
        attrs.iter().filter(|a| a.path().is_ident("doc")).collect();

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

    // Trait-bound name of the first non-hw dtype type parameter (e.g. `Float`),
    // used to infer the implicit "all dtypes" set when `dtypes` is omitted.
    let dtype_param_bound: Option<String> = input
        .sig
        .generics
        .params
        .iter()
        .find_map(|p| match p {
            GenericParam::Type(tp) if tp.ident != hw_ident => Some(tp),
            _ => None,
        })
        .and_then(|tp| {
            tp.bounds.iter().find_map(|b| {
                if let TypeParamBound::Trait(tb) = b {
                    tb.path.segments.last().map(|s| s.ident.to_string())
                } else {
                    None
                }
            })
        });

    // 3a. Collect const generic params — these become struct fields, not type params.
    let const_params: Vec<syn::ConstParam> = input
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let GenericParam::Const(cp) = p {
                Some(cp.clone())
            } else {
                None
            }
        })
        .collect();

    // Lowercased field idents for each const param (idiomatic Rust field naming).
    let const_field_idents: Vec<Ident> = const_params
        .iter()
        .map(|cp| format_ident!("{}", cp.ident.to_string().to_lowercase()))
        .collect();

    // 3b. Collect non-hw, non-const type params for the struct definition/usage.
    //     Const params are excluded: they become runtime fields instead.
    let struct_gen_params: Vec<&GenericParam> = input
        .sig
        .generics
        .params
        .iter()
        .filter(|p| match p {
            GenericParam::Type(tp) => tp.ident != hw_ident,
            GenericParam::Const(_) => false,
            GenericParam::Lifetime(_) => true,
        })
        .collect();

    let struct_gen_args: Vec<TokenStream2> = struct_gen_params
        .iter()
        .map(|p| match p {
            GenericParam::Type(tp) => {
                let i = &tp.ident;
                quote!(#i)
            }
            GenericParam::Lifetime(lp) => {
                let l = &lp.lifetime;
                quote!(#l)
            }
            GenericParam::Const(_) => unreachable!("const params are filtered above"),
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
            if let GenericParam::Type(tp) = p
                && tp.ident != hw_ident
            {
                let var = format_ident!("__type_name_{}", tp.ident.to_string().to_lowercase());
                return Some((tp.ident.clone(), var));
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
            if let FnArg::Typed(pt) = a {
                Some(pt)
            } else {
                None
            }
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
            let line = format!("let {name} = LlvmPointer({name} as *mut _);");
            quote! { ::std::string::String::from(#line) }
        })
        .collect();

    // Call arguments string (just the names, joined).
    let call_args_str: String = fn_inputs
        .iter()
        .map(|pt| pat_to_str(&pt.pat))
        .collect::<Vec<_>>()
        .join(", ");

    // Call type-arg expressions, one per original generic.
    // HW type → "LlvmTriton", dtype type params → runtime type name,
    // const params → the runtime field value (constructor argument).
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
                // Look up the lowercased field ident for this const param.
                let pos = const_params
                    .iter()
                    .position(|c| c.ident == cp.ident)
                    .expect("const param must exist in const_params");
                let field_ident = &const_field_idents[pos];
                quote! { (#field_ident).to_string() }
            }
            GenericParam::Lifetime(_) => quote! { ::std::string::String::new() },
        })
        .collect();

    // Original function source stored verbatim.
    let original_source_str = quote!(#vis #sig #block).to_string();

    // Struct ident (PascalCase of the function name).
    let struct_ident = Ident::new(&to_pascal_case(&fn_name_str), fn_ident.span());

    // 6. Const field definitions for the struct, and constructor parameter list.
    let const_field_defs: Vec<TokenStream2> = const_params
        .iter()
        .zip(const_field_idents.iter())
        .map(|(cp, field_name)| {
            let ty = &cp.ty;
            quote! {
                /// Compile-time kernel constant, from the annotated fn's `const` generics.
                pub #field_name: #ty,
            }
        })
        .collect();

    let const_constructor_args: Vec<TokenStream2> = const_params
        .iter()
        .zip(const_field_idents.iter())
        .map(|(cp, field_name)| {
            let ty = &cp.ty;
            quote! { #field_name: #ty }
        })
        .collect();

    // 7. ID parts: fn_name + runtime dtype names + const field values.
    //    Produces a human-readable string like "vector_add__f32__1024".
    let id_part_exprs: Vec<TokenStream2> = {
        let mut parts = vec![quote! { ::std::string::String::from(#fn_name_str) }];
        for (_, var) in &type_param_vars {
            parts.push(quote! { ::std::string::String::from(#var) });
        }
        for field_ident in &const_field_idents {
            parts.push(quote! { (#field_ident).to_string() });
        }
        parts
    };

    // 8. PhantomData to satisfy the "type parameter never used" requirement.
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

    // The PTX symbol name: "{fn_name}_entry_point", computed at macro-expansion time
    // so it can be embedded as a string literal in the generated concat!/format! call.
    let entry_point_fn_name = format!("{}_entry_point", fn_name_str);

    let struct_stream: TokenStream2 = quote! {
        #(#doc_attrs)*
        pub struct #struct_ident #struct_generics_def {
            /// The kernel function's name (e.g. `"flash_attention2_forward"`).
            pub name: &'static str,
            /// Unique kernel identifier: fn_name + dtype(s) + const values joined by "__".
            pub id: ::std::string::String,
            #(#const_field_defs)*
            /// The original kernel function source.
            pub kernel_source: ::std::string::String,
            /// The Rust source of the generated C-ABI entry-point wrapper function.
            pub entry_point_source: ::std::string::String,
            /// Combined source (`kernel_source + "\n\n" + entry_point_source`); used by the `Kernel` trait.
            pub source: ::std::string::String,
            #phantom_field
        }

        impl #struct_generics_def #struct_ident #struct_generics_use {
            /// Constructs a new kernel instance for these compile-time parameters.
            pub fn new( #(#const_constructor_args,)* ) -> Self {
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
                        "use triton::llvm::triton::pointer::LlvmPointer;\n",
                        "type LlvmTriton = triton::llvm::triton::LlvmTriton;\n",
                        "\n",
                        "#[no_mangle]\n",
                        "pub extern \"C\" fn ", #entry_point_fn_name, "({params}) {{\n",
                        "{body}\n",
                        "}}"
                    ),
                    params = __entry_params_str,
                    body = __body,
                );

                let __id = {
                    let __id_parts: ::std::vec::Vec<::std::string::String> =
                        vec![ #(#id_part_exprs),* ];
                    __id_parts.join("__")
                };

                let __kernel_source = ::std::string::String::from(__original_source);
                let __source = format!("{}\n\n{}", __kernel_source, __entry_point);
                Self {
                    name: #fn_name_str,
                    id: __id,
                    #(#const_field_idents,)*
                    kernel_source: __kernel_source,
                    entry_point_source: __entry_point,
                    source: __source,
                    #phantom_init
                }
            }
        }

        impl #struct_generics_def teeny_core::device::program::Kernel
            for #struct_ident #struct_generics_use
        {
            type Args<'__a> = ( #(#args_types,)* );

            fn id(&self) -> ::std::string::String {
                self.id.clone()
            }

            fn name(&self) -> &str {
                self.name
            }

            fn source(&self) -> &str {
                &self.source
            }

            fn kernel_source(&self) -> &str {
                &self.kernel_source
            }

            fn entry_point_source(&self) -> &str {
                &self.entry_point_source
            }
        }
    };

    // 9. Optional dtype dispatcher, generated when the kernel opts into dispatch
    //    via `#[kernel(dtypes = [..])]` and/or `#[kernel(backward = ..)]`. Maps a
    //    runtime `DtypeRepr` to the monomorphized kernel struct (and its paired
    //    backward, if declared), returning a crate-agnostic `KernelInstance`.
    //
    //    Effective dtypes: the explicit `dtypes` list if given, otherwise (when
    //    dispatch is opted into via `backward`) every dtype permitted by the
    //    dtype type-parameter's trait bound — "no dtypes specified ⇒ all dtypes".
    let effective_dtypes: Vec<Ident> = if !kernel_attrs.dtypes.is_empty() {
        kernel_attrs.dtypes.clone()
    } else if kernel_attrs.backward.is_some() {
        match dtype_param_bound.as_deref().and_then(all_dtypes_for_bound) {
            Some(names) => names
                .iter()
                .map(|n| Ident::new(n, fn_ident.span()))
                .collect(),
            None => {
                return syn::Error::new_spanned(
                    &fn_ident,
                    "cannot infer supported dtypes: a `#[kernel]` that opts into dispatch \
                     without an explicit `dtypes = [..]` must have a dtype type parameter \
                     bound by one of Dtype/Num/Int/Float/Bool",
                )
                .to_compile_error()
                .into();
            }
        }
    } else {
        Vec::new()
    };

    let dispatcher_stream: TokenStream2 = if effective_dtypes.is_empty() {
        quote! {}
    } else {
        let dispatch_ident = format_ident!("{}Dispatch", struct_ident);
        let has_type_param = !type_param_vars.is_empty();
        let reprs: Vec<TokenStream2> = effective_dtypes
            .iter()
            .map(|dt| dtype_ident_to_repr(dt).expect("validated set"))
            .collect();
        let backward = kernel_attrs.backward.clone();

        let arms: Vec<TokenStream2> = effective_dtypes
            .iter()
            .map(|dt| {
                let repr = dtype_ident_to_repr(dt).expect("validated in parse_kernel_attrs");
                let concrete = dt;
                let fwd_new = if has_type_param {
                    quote! { #struct_ident::<#concrete>::new( #(#const_field_idents),* ) }
                } else {
                    quote! { #struct_ident::new( #(#const_field_idents),* ) }
                };
                let backward_expr = if let Some(bwd) = &backward {
                    let bwd_new = if has_type_param {
                        quote! { #bwd::<#concrete>::new( #(#const_field_idents),* ) }
                    } else {
                        quote! { #bwd::new( #(#const_field_idents),* ) }
                    };
                    quote! {{
                        let __b = #bwd_new;
                        ::core::option::Option::Some(teeny_core::model::KernelInstanceBackward {
                            name: __b.name.to_string(),
                            source: __b.source.clone(),
                        })
                    }}
                } else {
                    quote! { ::core::option::Option::None }
                };
                quote! {
                    #repr => {
                        let __f = #fwd_new;
                        teeny_core::model::KernelInstance {
                            name: __f.name.to_string(),
                            source: __f.source.clone(),
                            runtime_op: ::std::sync::Arc::new(__f),
                            backward: #backward_expr,
                        }
                    }
                }
            })
            .collect();

        let fn_name_for_err = fn_name_str.clone();
        quote! {
            #(#doc_attrs)*
            pub struct #dispatch_ident;

            impl #dispatch_ident {
                /// Dtypes this kernel declares support for.
                pub const SUPPORTED_DTYPES: &'static [teeny_core::graph::DtypeRepr] =
                    &[ #(#reprs),* ];

                /// Instantiate the kernel for a runtime `dtype`, returning a
                /// `KernelInstance` (forward + optional backward). Errors for
                /// any dtype outside [`SUPPORTED_DTYPES`].
                #[allow(clippy::too_many_arguments)]
                pub fn dispatch(
                    dtype: teeny_core::graph::DtypeRepr,
                    #(#const_constructor_args,)*
                ) -> ::anyhow::Result<teeny_core::model::KernelInstance> {
                    ::core::result::Result::Ok(match dtype {
                        #(#arms)*
                        other => {
                            return ::core::result::Result::Err(::anyhow::anyhow!(
                                "{} does not support dtype {:?} (supported: {:?})",
                                #fn_name_for_err, other, Self::SUPPORTED_DTYPES
                            ));
                        }
                    })
                }
            }
        }
    };

    let mut result: TokenStream = TokenStream::from(function_stream);
    result.extend(TokenStream::from(struct_stream));
    result.extend(TokenStream::from(dispatcher_stream));
    result
}
