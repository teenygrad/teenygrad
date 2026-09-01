/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! The `#[tiled_kernel]` macro: [`super::kernel::kernel`] plus a few
//! `#[tiled_kernel]`-specific codegen bits (dtype dispatch, a permanently-
//! `None` `fusion_core()` stub). Its old attribute DSL --
//! `#[tile(...)]`/`#[tile_loop(...)]`/`#[tile_pid_swizzle(...)]` and the
//! index-arithmetic prelude codegen they drove (single-axis load prelude,
//! GEMM's swizzled `pid` decode) -- was removed outright at `84ca6eedf`:
//! that codegen baked index arithmetic into the individual kernel function
//! being compiled, which doesn't compose when a kernel is called as a
//! tile-op from another kernel's body (index arithmetic belongs in a
//! wrapper, not each composed function) -- see teenygrad-1nr.1.
//!
//! A narrower `#[tile(block=..,extent=..)]` was revived on top of the
//! `In<Tile<HW,D>>`/`Out<Tile<HW,D>>` auto-prelude (teenygrad-1nr.1's own
//! addition, `c69c08b63`) by teenygrad-1nr.18, scoped to avoid repeating
//! `84ca6eedf`'s mistake: it drives *only* the auto-prelude's own
//! block/extent naming (still the single flat axis
//! `arange(block)+pid*block` shape that prelude already had -- this does
//! not add N-axis/windowed/looped prelude codegen) and this file's
//! generated `tile_spec()` method. It never re-splices index arithmetic
//! into the kernel author's own body. Optional and additive: a `Tile`
//! parameter with no `#[tile(...)]` falls back to the pre-existing
//! hardcoded `BLOCK_SIZE`/`n_elements` convention, unchanged, and no
//! `tile_spec()` is generated for it. See [`super::common`] for the
//! parsing/codegen helpers shared with the plain `#[kernel]` macro.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, FnArg, GenericParam, Ident, ItemFn, MetaNameValue, Pat, PatType, Token, Type,
    TypeParamBound, parse::Parser, parse_macro_input, punctuated::Punctuated,
};

use super::common::{
    PtrArgKind, all_dtypes_for_bound, classify_pointer_arg, dtype_ident_to_repr,
    extract_pointer_inner, in_tile_dtype, out_tile_dtype, parse_kernel_attrs, pat_to_str,
    rewrite_tile_param_to_pointer, simple_type_ident, to_pascal_case, unwrap_pointer_marker,
};

/// Parsed `#[tile(...)]` on one `In<Tile<..>>`/`Out<Tile<..>>` parameter
/// -- or, since teenygrad-1nr.19, on any `In`/`Out`/`InOut`-marked
/// parameter, `Tile`-wrapped or a raw pointer -- describing one axis of
/// that parameter's tensor (teenygrad-1nr.18/teenygrad-1nr.19).
///
/// A parameter may carry more than one `#[tile(...)]` attribute
/// (repeatable, like `#[doc = "..."]`): each occurrence is one axis, in
/// declaration order (outermost first -- matches this codebase's existing
/// `TensorTileSpec`/hand-authored `KernelTileSpec` convention of dim 0 =
/// outermost, e.g. NCHW's `B`). Exactly one occurrence on a `Tile`-typed
/// parameter is the original teenygrad-1nr.18 shape (drives the
/// auto-prelude too, see [`parse_tile_attrs`]'s caller); more than one, or
/// any occurrence at all on a raw-pointer parameter, is purely
/// declarative -- teenygrad-1nr.19 -- and never touches codegen.
struct TileAttrArgs {
    /// `Some(block_const)` when this axis is block-tiled (one CTA covers
    /// `block_const` elements); `None` for an untiled axis (one CTA per
    /// index). Required (`Some`) on a `Tile`-typed parameter -- untiled
    /// `Tile` axes aren't supported yet, see the auto-prelude's own
    /// requirements.
    block: Option<Ident>,
    /// The `{NAME}: i32` parameter this axis's extent is read from.
    extent: Ident,
    /// This axis's identity for [`::teeny_core::model::GridSpec`]
    /// matching (teenygrad-1nr.19) -- defaults to `extent`'s own name
    /// when omitted.
    name: Option<syn::LitStr>,
    /// Which real hardware grid dimension this axis reads from
    /// (teenygrad-1nr.19) -- `X`, `Y`, or `Z`; defaults to `X`.
    dim: Option<Ident>,
}

/// Parse one `#[tile(...)]` attribute's contents.
fn parse_one_tile_attr(attr: &syn::Attribute) -> Result<TileAttrArgs, syn::Error> {
    let meta_list = attr.meta.require_list()?;
    let parsed = Punctuated::<MetaNameValue, Token![,]>::parse_terminated
        .parse2(meta_list.tokens.clone())?;
    let mut block = None;
    let mut extent = None;
    let mut name = None;
    let mut dim = None;
    for nv in parsed {
        let key = nv
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        if key == "name" {
            let Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Str(s),
                ..
            }) = &nv.value
            else {
                return Err(syn::Error::new_spanned(
                    &nv.value,
                    "`#[tile(name = ..)]` must be a string literal",
                ));
            };
            name = Some(s.clone());
            continue;
        }
        let Expr::Path(p) = &nv.value else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                "`#[tile(...)]` values must be bare identifiers (except `name`, a string literal)",
            ));
        };
        let Some(id) = p.path.get_ident().cloned() else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                "`#[tile(...)]` values must be a single identifier",
            ));
        };
        match key.as_str() {
            "block" => block = Some(id),
            "extent" => extent = Some(id),
            "dim" => {
                if !matches!(id.to_string().as_str(), "X" | "Y" | "Z") {
                    return Err(syn::Error::new_spanned(
                        &id,
                        "`#[tile(dim = ..)]` must be `X`, `Y`, or `Z`",
                    ));
                }
                dim = Some(id);
            }
            other => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    format!(
                        "unknown `#[tile(...)]` key `{other}` (expected `block`, `extent`, \
                         `name`, or `dim`)"
                    ),
                ));
            }
        }
    }
    let extent = extent
        .ok_or_else(|| syn::Error::new_spanned(attr, "`#[tile(...)]` requires `extent = ..`"))?;
    Ok(TileAttrArgs {
        block,
        extent,
        name,
        dim,
    })
}

/// Parse every `#[tile(...)]` attribute on one parameter, in declaration
/// order.
fn parse_tile_attrs(pt: &PatType) -> Result<Vec<TileAttrArgs>, syn::Error> {
    pt.attrs
        .iter()
        .filter(|a| a.path().is_ident("tile"))
        .map(parse_one_tile_attr)
        .collect()
}

/// Strip a `#[tile(...)]` attribute from a parameter's attribute list, if
/// present -- it is host-only metadata (like `Tile` itself), never a real
/// attribute macro registered anywhere, so it must not reach the
/// regenerated device/host signatures this macro emits.
fn strip_tile_attr(pt: &mut PatType) {
    pt.attrs.retain(|a| !a.path().is_ident("tile"));
}

// ── Macro implementation ──────────────────────────────────────────────────────

pub fn tiled_kernel(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let kernel_attrs = match parse_kernel_attrs(attrs) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let input = parse_macro_input!(item as ItemFn);
    let fn_ident = input.sig.ident.clone();
    let fn_name_str = fn_ident.to_string();
    let vis = &input.vis;
    let attrs: Vec<&syn::Attribute> = input.attrs.iter().collect();
    let attrs = &attrs;
    let sig = &input.sig;

    // Doc comments (`#[doc = "..."]`, from `///`/`//!`) on the annotated fn,
    // forwarded onto the generated struct(s) below -- they're the actual
    // public item downstream users and rustdoc see, so without this the
    // fn's docs never reach anything `missing_docs` checks.
    let doc_attrs: Vec<&syn::Attribute> = attrs
        .iter()
        .copied()
        .filter(|a| a.path().is_ident("doc"))
        .collect();

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
        .expect("#[tiled_kernel] requires a type parameter with a `Triton` bound");

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

    // Pointer args must be In / Out / InOut (required for KernelIo / fusion).
    for pt in &fn_inputs {
        if let Some((PtrArgKind::Raw, _)) = classify_pointer_arg(&pt.ty, &hw_ident) {
            let name = pat_to_str(&pt.pat);
            return syn::Error::new_spanned(
                &pt.ty,
                format!(
                    "pointer argument `{name}` must be wrapped in `In` / `Out` / \
                     `InOut` so fusion can classify I/O by signature"
                ),
            )
            .to_compile_error()
            .into();
        }
    }

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

    // Pointer-wrapping lines for the entry point.
    // Device fns take bare pointers (markers stripped below); wrap as LlvmPointer only.
    let ptr_conv_exprs: Vec<TokenStream2> = fn_inputs
        .iter()
        .filter_map(|pt| {
            let _ = classify_pointer_arg(&pt.ty, &hw_ident)?;
            let name = pat_to_str(&pt.pat);
            let line = format!("let {name} = LlvmPointer({name} as *mut _);");
            Some(quote! { ::std::string::String::from(#line) })
        })
        .collect();

    // Pointer roles in signature order for KernelIo (scalars omitted).
    let ptr_roles: Vec<TokenStream2> = fn_inputs
        .iter()
        .filter_map(|pt| {
            let (kind, _) = classify_pointer_arg(&pt.ty, &hw_ident)?;
            let role = match kind {
                PtrArgKind::In => quote! { ::teeny_triton::PtrRole::In },
                PtrArgKind::Out => quote! { ::teeny_triton::PtrRole::Out },
                PtrArgKind::InOut => quote! { ::teeny_triton::PtrRole::InOut },
                PtrArgKind::Raw => quote! { ::teeny_triton::PtrRole::Raw },
            };
            Some(role)
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

    // Auto-generated prelude for `In<Tile<HW, D>>` / `Out<Tile<HW, D>>`
    // parameters (teenygrad-1nr.1): unlike the removed `#[tile(...)]` DSL,
    // this is driven by the parameter *type*, not a separate attribute, and
    // the prelude is spliced ahead of the kernel author's own body rather
    // than replacing it -- `Tile` never crosses the device/host ABI (see
    // `common::unwrap_pointer_marker`), so the parameter name is shadowed by
    // a `Tile` value via an ordinary `let`. `In` params are shadowed by a
    // *loaded* tile (`.tensor` is `HW::load(...)`); `Out` params are
    // shadowed by an *addressed* tile (`.tensor` is the offset write
    // pointer, not a loaded value) so the kernel body can call
    // `HW::store(y.tensor, value, y.mask, ...)` without ever calling
    // `.add_offsets` itself.
    let tile_attrs: Vec<Vec<TileAttrArgs>> = match fn_inputs
        .iter()
        .map(|pt| parse_tile_attrs(pt))
        .collect::<Result<Vec<_>, syn::Error>>()
    {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    // teenygrad-1nr.19: a `Tile`-typed parameter's auto-prelude below only
    // understands one flat axis -- more than one `#[tile(...)]` on such a
    // parameter would need N-axis prelude codegen this macro doesn't have.
    // Metadata-only, multi-axis declarations belong on a raw pointer
    // parameter instead (see `structured_params` below), which the
    // prelude never touches.
    for (pt, attrs) in fn_inputs.iter().zip(tile_attrs.iter()) {
        let is_tile = in_tile_dtype(&pt.ty, &hw_ident).is_some()
            || out_tile_dtype(&pt.ty, &hw_ident).is_some();
        if is_tile && attrs.len() > 1 {
            return syn::Error::new_spanned(
                pt,
                "multi-axis `#[tile(...)]` on an `In<Tile<..>>`/`Out<Tile<..>>` parameter isn't \
                 supported yet -- the auto-prelude only understands one flat axis \
                 (teenygrad-1nr.18). Annotate a raw pointer parameter instead for \
                 metadata-only, multi-axis declarations (teenygrad-1nr.19).",
            )
            .to_compile_error()
            .into();
        }
    }

    let tile_in_params: Vec<(&Ident, Type, Option<&TileAttrArgs>)> = fn_inputs
        .iter()
        .zip(tile_attrs.iter())
        .filter_map(|(pt, attrs)| {
            let dtype = in_tile_dtype(&pt.ty, &hw_ident)?;
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            Some((&pi.ident, dtype, attrs.first()))
        })
        .collect();
    let tile_out_params: Vec<(&Ident, Type, Option<&TileAttrArgs>)> = fn_inputs
        .iter()
        .zip(tile_attrs.iter())
        .filter_map(|(pt, attrs)| {
            let dtype = out_tile_dtype(&pt.ty, &hw_ident)?;
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            Some((&pi.ident, dtype, attrs.first()))
        })
        .collect();

    // teenygrad-1nr.19: raw-pointer `In`/`Out`/`InOut` parameters (never
    // `Tile`-typed -- those are handled above) carrying one or more
    // `#[tile(...)]` attributes get metadata-only `tile_spec()`/
    // `grid_spec()` generation. No prelude, no change to the kernel body
    // at all -- unlike the `Tile`-typed case, these parameters were
    // already hand-indexed by the kernel author (e.g. `conv2d_forward`'s
    // own `pid` decode), and stay that way. Each attribute is one real
    // tensor axis, in declaration order (outermost first, matching this
    // codebase's existing dim-0-is-outermost convention).
    let structured_params: Vec<(&Ident, PtrArgKind, &[TileAttrArgs])> = fn_inputs
        .iter()
        .zip(tile_attrs.iter())
        .filter_map(|(pt, attrs)| {
            if attrs.is_empty()
                || in_tile_dtype(&pt.ty, &hw_ident).is_some()
                || out_tile_dtype(&pt.ty, &hw_ident).is_some()
            {
                return None;
            }
            let (kind, _) = classify_pointer_arg(&pt.ty, &hw_ident)?;
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            Some((&pi.ident, kind, attrs.as_slice()))
        })
        .collect();

    for (_, _, axes) in &structured_params {
        for axis in *axes {
            if let Some(block) = &axis.block
                && !const_params.iter().any(|cp| &cp.ident == block)
            {
                return syn::Error::new_spanned(
                    block,
                    format!(
                        "`#[tile(block = {block})]` names a const generic this kernel doesn't \
                         declare"
                    ),
                )
                .to_compile_error()
                .into();
            }
            let extent = &axis.extent;
            let extent_ok = fn_inputs.iter().any(|pt| {
                let name_ok = matches!(&*pt.pat, Pat::Ident(pi) if &pi.ident == extent);
                let ty_ok = matches!(&*pt.ty, Type::Path(tp) if tp.path.is_ident("i32"));
                name_ok && ty_ok
            });
            if !extent_ok {
                return syn::Error::new_spanned(
                    extent,
                    format!(
                        "`#[tile(extent = {extent})]` names a parameter this kernel doesn't \
                         declare as `{extent}: i32`"
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    }

    // teenygrad-1nr.18: resolve the one (block, extent) axis pair the
    // auto-prelude below uses. Prefer an explicit `#[tile(block=..,
    // extent=..)]` when *every* tile-typed parameter on this kernel
    // declares it (and they all agree on the same pair -- per-parameter/
    // multi-axis attributes aren't supported by the auto-prelude yet);
    // otherwise fall back to the pre-existing hardcoded `BLOCK_SIZE`/
    // `n_elements` convention, unchanged. `tile_spec()` (below) is only
    // generated in the explicit case.
    let all_tile_param_attrs: Vec<Option<&TileAttrArgs>> = tile_in_params
        .iter()
        .chain(tile_out_params.iter())
        .map(|(_, _, a)| *a)
        .collect();
    let has_explicit_tile_attr =
        !all_tile_param_attrs.is_empty() && all_tile_param_attrs.iter().all(Option::is_some);
    if !all_tile_param_attrs.is_empty()
        && all_tile_param_attrs.iter().any(Option::is_some)
        && !has_explicit_tile_attr
    {
        return syn::Error::new_spanned(
            &input.sig,
            "when any `In<Tile<..>>`/`Out<Tile<..>>` parameter on this kernel declares \
             `#[tile(block=..,extent=..)]`, every such parameter must declare it",
        )
        .to_compile_error()
        .into();
    }
    if has_explicit_tile_attr {
        let mut iter = all_tile_param_attrs
            .iter()
            .map(|a| a.expect("checked above"));
        let first = iter.next().expect("non-empty, checked above");
        if first.block.is_none() {
            return syn::Error::new_spanned(
                &first.extent,
                "an `In<Tile<..>>`/`Out<Tile<..>>` parameter's `#[tile(...)]` requires \
                 `block = ..` -- untiled `Tile` axes aren't supported yet",
            )
            .to_compile_error()
            .into();
        }
        for other in iter {
            if other.block != first.block || other.extent != first.extent {
                return syn::Error::new_spanned(
                    other.block.as_ref().unwrap_or(&other.extent),
                    "all `In<Tile<..>>`/`Out<Tile<..>>` parameters on one kernel must share \
                     the same `#[tile(block=..,extent=..)]` axis today -- per-parameter axes \
                     aren't supported by the auto-prelude yet (teenygrad-1nr.18)",
                )
                .to_compile_error()
                .into();
            }
        }
    }

    let final_block = if tile_in_params.is_empty() && tile_out_params.is_empty() {
        input.block.as_ref().clone()
    } else {
        let (block_ident, extent_ident): (Ident, Ident) = if has_explicit_tile_attr {
            let args = all_tile_param_attrs[0].expect("checked above");
            let block = args
                .block
                .clone()
                .expect("checked above: block is required on a Tile-typed parameter");
            let extent = args.extent.clone();
            if !const_params.iter().any(|cp| cp.ident == block) {
                return syn::Error::new_spanned(
                    &block,
                    format!(
                        "`#[tile(block = {block})]` names a const generic this kernel doesn't \
                         declare"
                    ),
                )
                .to_compile_error()
                .into();
            }
            let extent_ok = fn_inputs.iter().any(|pt| {
                let name_ok = matches!(&*pt.pat, Pat::Ident(pi) if pi.ident == extent);
                let ty_ok = matches!(&*pt.ty, Type::Path(tp) if tp.path.is_ident("i32"));
                name_ok && ty_ok
            });
            if !extent_ok {
                return syn::Error::new_spanned(
                    &extent,
                    format!(
                        "`#[tile(extent = {extent})]` names a parameter this kernel doesn't \
                         declare as `{extent}: i32`"
                    ),
                )
                .to_compile_error()
                .into();
            }
            (block, extent)
        } else {
            let block_size = const_params.iter().find(|cp| cp.ident == "BLOCK_SIZE");
            let Some(block_size) = block_size else {
                return syn::Error::new_spanned(
                    &input.sig,
                    "an `In<Tile<..>>`/`Out<Tile<..>>` parameter requires this kernel to \
                     declare `const BLOCK_SIZE: i32` (or an explicit \
                     `#[tile(block=..,extent=..)]`)",
                )
                .to_compile_error()
                .into();
            };
            let has_n_elements = fn_inputs.iter().any(|pt| {
                let name_ok = matches!(&*pt.pat, Pat::Ident(pi) if pi.ident == "n_elements");
                let ty_ok = matches!(&*pt.ty, Type::Path(tp) if tp.path.is_ident("i32"));
                name_ok && ty_ok
            });
            if !has_n_elements {
                return syn::Error::new_spanned(
                    &input.sig,
                    "an `In<Tile<..>>`/`Out<Tile<..>>` parameter requires this kernel to \
                     declare an `n_elements: i32` parameter (or an explicit \
                     `#[tile(block=..,extent=..)]`)",
                )
                .to_compile_error()
                .into();
            }
            (block_size.ident.clone(), format_ident!("n_elements"))
        };

        let mut stmts: Vec<syn::Stmt> = syn::parse2::<syn::Block>(quote! {{
            let pid = #hw_ident::program_id(Axis::X);
            let block_start = pid * #block_ident;
            let offsets = #hw_ident::arange(0, #block_ident) + block_start;
            let in_bounds = offsets.lt(#extent_ident);
        }})
        .expect("generated tile prelude is valid Rust")
        .stmts;
        for (ident, dtype, _) in &tile_in_params {
            let load_stmt: syn::Stmt = syn::parse2(quote! {
                let #ident = Tile::<#hw_ident, #dtype> {
                    tensor: #hw_ident::load(
                        #ident.add_offsets(offsets),
                        Some(in_bounds),
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    ),
                    mask: Some(in_bounds),
                };
            })
            .expect("generated tile load statement is valid Rust");
            stmts.push(load_stmt);
        }
        for (ident, dtype, _) in &tile_out_params {
            // `.add_offsets()` returns `HW::Tensor<HW::Pointer<D>>` (a tensor
            // of write addresses), not `HW::Tensor<D>` (a tensor of `D`
            // values) -- so the shadowed `Tile` is instantiated with
            // `HW::Pointer<D>` as its own dtype param, not `D` itself.
            let addr_stmt: syn::Stmt = syn::parse2(quote! {
                let #ident = Tile::<#hw_ident, #hw_ident::Pointer<#dtype>> {
                    tensor: #ident.add_offsets(offsets),
                    mask: Some(in_bounds),
                };
            })
            .expect("generated tile address statement is valid Rust");
            stmts.push(addr_stmt);
        }
        stmts.extend(input.block.stmts.iter().cloned());
        syn::Block {
            brace_token: input.block.brace_token,
            stmts,
        }
    };

    // teenygrad-1nr.18: when every tile-typed parameter carries an explicit
    // `#[tile(block=..,extent=..)]`, emit a `tile_spec()` method built from
    // that same attribute data instead of requiring a hand-authored
    // `KernelTileSpec` at the `TritonLowering` call site. `rank` is a
    // runtime argument, not baked in here: the real tensor rank is a
    // property of the graph node this kernel gets applied to, not of the
    // kernel's own signature (a `Tile`-typed param carries a dtype, never a
    // rank) -- same reasoning as `teeny-kernels`'
    // `flat_elementwise_tile_spec`, which this method mirrors the shape of.
    if has_explicit_tile_attr && !structured_params.is_empty() {
        return syn::Error::new_spanned(
            &input.sig,
            "this kernel mixes `#[tile(...)]`-tagged `In<Tile<..>>`/`Out<Tile<..>>` parameters \
             with `#[tile(...)]`-tagged raw pointer parameters -- not supported in one kernel \
             yet (teenygrad-1nr.19)",
        )
        .to_compile_error()
        .into();
    }

    let (tile_spec_method, grid_spec_method): (TokenStream2, TokenStream2) =
        if has_explicit_tile_attr {
            let args = all_tile_param_attrs[0].expect("checked above");
            let block_str = args
                .block
                .as_ref()
                .expect("checked above: block is required on a Tile-typed parameter")
                .to_string();
            let extent_str = args.extent.to_string();
            let in_param_strs: Vec<String> = tile_in_params
                .iter()
                .map(|(id, _, _)| id.to_string())
                .collect();
            let out_param_strs: Vec<String> = tile_out_params
                .iter()
                .map(|(id, _, _)| id.to_string())
                .collect();
            let tile_spec = quote! {
                /// Declarative tile-shape metadata derived from this kernel's
                /// `#[tile(block=..,extent=..)]`-tagged `In<Tile<..>>`/
                /// `Out<Tile<..>>` parameters (teenygrad-1nr.18).
                pub fn tile_spec(rank: usize) -> ::teeny_core::model::KernelTileSpec {
                    let dims: &'static [usize] = ::std::boxed::Box::leak(
                        (0..rank).collect::<::std::vec::Vec<usize>>().into_boxed_slice(),
                    );
                    let axes: &'static [::teeny_core::model::TileAxisBinding] =
                        ::std::boxed::Box::leak(::std::boxed::Box::new([
                            ::teeny_core::model::TileAxisBinding {
                                dims,
                                block_const: #block_str,
                                extent_param: #extent_str,
                                window: ::core::option::Option::None,
                                divide_by: ::core::option::Option::None,
                            },
                        ]));
                    let inputs: &'static [::teeny_core::model::TensorTileSpec] =
                        ::std::boxed::Box::leak(::std::boxed::Box::new([ #(
                            ::teeny_core::model::TensorTileSpec {
                                param: #in_param_strs,
                                rank,
                                axes,
                                reduction_axis: ::core::option::Option::None,
                                untiled_dims: &[],
                            }
                        ),* ]));
                    let outputs: &'static [::teeny_core::model::TensorTileSpec] =
                        ::std::boxed::Box::leak(::std::boxed::Box::new([ #(
                            ::teeny_core::model::TensorTileSpec {
                                param: #out_param_strs,
                                rank,
                                axes,
                                reduction_axis: ::core::option::Option::None,
                                untiled_dims: &[],
                            }
                        ),* ]));
                    ::teeny_core::model::KernelTileSpec {
                        inputs,
                        outputs,
                        loop_spec: ::core::option::Option::None,
                    }
                }
            };
            // teenygrad-1nr.19: the flat/single-axis case always has exactly
            // one grid axis (the whole flattened tensor), regardless of the
            // real tensor's rank -- unlike `tile_spec()` above, no runtime
            // `rank` argument is needed here.
            let grid_spec = quote! {
                /// Declarative launch-grid metadata derived from the same
                /// `#[tile(block=..,extent=..)]` attribute as `tile_spec()`
                /// (teenygrad-1nr.19).
                pub fn grid_spec() -> ::teeny_core::model::GridSpec {
                    ::teeny_core::model::GridSpec {
                        axes: &[
                            ::teeny_core::model::GridAxisBinding {
                                name: #extent_str,
                                extent_factors: &[#extent_str, #block_str],
                                dim: ::teeny_core::model::GridDim::X,
                                block_const: ::core::option::Option::Some(#block_str),
                            },
                        ],
                    }
                }
            };
            (tile_spec, grid_spec)
        } else if !structured_params.is_empty() {
            // teenygrad-1nr.19: metadata-only `tile_spec()`/`grid_spec()` for
            // raw-pointer parameters, generated straight from their
            // `#[tile(...)]` axis declarations -- see `structured_params`'s
            // own comment above. Every axis count is known at macro-expansion
            // time (one `#[tile(...)]` occurrence per real tensor dim), so
            // unlike the flat/single-axis case above, neither method needs a
            // runtime argument, and nothing needs `Box::leak` -- the axis
            // arrays are ordinary `&'static` literals.
            let mut input_tensor_specs: Vec<TokenStream2> = Vec::new();
            let mut output_tensor_specs: Vec<TokenStream2> = Vec::new();
            let mut grid_output: Option<(&Ident, &[TileAttrArgs])> = None;
            for (ident, kind, axes) in &structured_params {
                let param_str = ident.to_string();
                let rank = axes.len();
                let mut tiled_axis_tokens: Vec<TokenStream2> = Vec::new();
                let mut untiled_name_tokens: Vec<String> = Vec::new();
                for (i, axis) in axes.iter().enumerate() {
                    match &axis.block {
                        Some(block) => {
                            let block_str = block.to_string();
                            let extent_str = axis.extent.to_string();
                            tiled_axis_tokens.push(quote! {
                                ::teeny_core::model::TileAxisBinding {
                                    dims: &[#i],
                                    block_const: #block_str,
                                    extent_param: #extent_str,
                                    window: ::core::option::Option::None,
                                    divide_by: ::core::option::Option::None,
                                }
                            });
                        }
                        None => {
                            let label = axis
                                .name
                                .as_ref()
                                .map(syn::LitStr::value)
                                .unwrap_or_else(|| axis.extent.to_string());
                            untiled_name_tokens.push(label);
                        }
                    }
                }
                let tensor_spec = quote! {
                    ::teeny_core::model::TensorTileSpec {
                        param: #param_str,
                        rank: #rank,
                        axes: &[ #(#tiled_axis_tokens),* ],
                        reduction_axis: ::core::option::Option::None,
                        untiled_dims: &[ #(#untiled_name_tokens),* ],
                    }
                };
                match kind {
                    PtrArgKind::In => input_tensor_specs.push(tensor_spec),
                    PtrArgKind::Out => {
                        output_tensor_specs.push(tensor_spec);
                        grid_output = Some((ident, axes));
                    }
                    PtrArgKind::InOut => {
                        output_tensor_specs.push(tensor_spec.clone());
                        input_tensor_specs.push(tensor_spec);
                        grid_output = Some((ident, axes));
                    }
                    PtrArgKind::Raw => unreachable!(
                        "every fn_input pointer arg is already required to be In/Out/InOut, \
                     checked earlier in this function"
                    ),
                }
            }
            let tile_spec = quote! {
                /// Declarative tile-shape metadata derived from this kernel's
                /// `#[tile(...)]`-tagged raw pointer parameters
                /// (teenygrad-1nr.19).
                pub fn tile_spec() -> ::teeny_core::model::KernelTileSpec {
                    ::teeny_core::model::KernelTileSpec {
                        inputs: &[ #(#input_tensor_specs),* ],
                        outputs: &[ #(#output_tensor_specs),* ],
                        loop_spec: ::core::option::Option::None,
                    }
                }
            };
            // Welder's own model: the fused group's boundary *output* edge's
            // shape drives the grid (teenygrad-1nr.17) -- so `grid_spec()` is
            // built from the one `Out`/`InOut` structured parameter's axes,
            // not every structured parameter's. Omitted entirely (no
            // `grid_spec()` generated) when that's ambiguous -- zero or more
            // than one qualifying parameter.
            let grid_spec = match grid_output {
                Some((_, axes)) => {
                    let axis_tokens: Vec<TokenStream2> = axes
                        .iter()
                        .map(|axis| {
                            let name = axis
                                .name
                                .as_ref()
                                .map(syn::LitStr::value)
                                .unwrap_or_else(|| axis.extent.to_string());
                            let extent_str = axis.extent.to_string();
                            let dim_variant =
                                match axis.dim.as_ref().map(ToString::to_string).as_deref() {
                                    Some("Y") => quote! { ::teeny_core::model::GridDim::Y },
                                    Some("Z") => quote! { ::teeny_core::model::GridDim::Z },
                                    _ => quote! { ::teeny_core::model::GridDim::X },
                                };
                            let (block_const_tok, extent_factors_tok) = match &axis.block {
                                Some(block) => {
                                    let block_str = block.to_string();
                                    (
                                        quote! { ::core::option::Option::Some(#block_str) },
                                        quote! { &[#extent_str, #block_str] },
                                    )
                                }
                                None => (
                                    quote! { ::core::option::Option::None },
                                    quote! { &[#extent_str] },
                                ),
                            };
                            quote! {
                                ::teeny_core::model::GridAxisBinding {
                                    name: #name,
                                    extent_factors: #extent_factors_tok,
                                    dim: #dim_variant,
                                    block_const: #block_const_tok,
                                }
                            }
                        })
                        .collect();
                    quote! {
                        /// Declarative launch-grid metadata derived from this
                        /// kernel's `Out`/`InOut` `#[tile(...)]`-tagged raw
                        /// pointer parameter (teenygrad-1nr.19). Welder's own
                        /// model: the fused group's boundary output drives
                        /// the grid.
                        pub fn grid_spec() -> ::teeny_core::model::GridSpec {
                            ::teeny_core::model::GridSpec {
                                axes: &[ #(#axis_tokens),* ],
                            }
                        }
                    }
                }
                None => quote! {},
            };
            (tile_spec, grid_spec)
        } else {
            (quote! {}, quote! {})
        };

    // FusionCore splice-body extraction (teenygrad-3w0.9) identified its
    // eligible kernels via `#[tile(...)]`'s tile_attrs, which no longer
    // exist -- see teenygrad-1nr.1. `fusion_core()` is unconditionally
    // `None` now; nothing computes a `Some(..)` for it any more.
    let fusion_core_body: TokenStream2 = quote! {
        pub fn fusion_core() -> ::core::option::Option<::teeny_triton::FusionCore> {
            ::core::option::Option::None
        }
    };

    // Device-side source: same body, but pointer markers stripped so MLIR sees
    // bare `T::Pointer<D>`. Host keeps the marked signature for KernelIo / API.
    // `#[tile(...)]` (teenygrad-1nr.18) is stripped from both regenerated
    // signatures below -- it's host-only metadata, like `Tile` itself, never
    // a real attribute macro registered anywhere.
    let mut device_sig = sig.clone();
    for input in device_sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = input {
            *pt.ty = unwrap_pointer_marker(&pt.ty, &hw_ident);
            strip_tile_attr(pt);
        }
    }
    let original_source_str = quote!(#vis #device_sig #final_block).to_string();

    // Host-side signature: same as the original, except `In<Tile<HW,D>>` /
    // `Out<Tile<HW,D>>` / `InOut<Tile<HW,D>>` params get their inner type
    // rewritten back to `HW::Pointer<D>` (marker kept) -- `Tile` never
    // crosses the real ABI, and `ptr_unwraps` below derefs through the
    // marker to a `HW::Pointer<D>` that the (possibly tile-prelude-bearing)
    // body expects, matching `device_sig`'s treatment of the same params.
    let mut host_sig = sig.clone();
    for input in host_sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = input {
            *pt.ty = rewrite_tile_param_to_pointer(&pt.ty, &hw_ident);
            strip_tile_attr(pt);
        }
    }

    // Host fn: unwrap In/Out markers at body start so descriptor/load/store APIs
    // see bare `T::Pointer` (by-value args do not autoderef through Deref).
    let ptr_unwraps: Vec<TokenStream2> = fn_inputs
        .iter()
        .filter_map(|pt| {
            let (kind, _) = classify_pointer_arg(&pt.ty, &hw_ident)?;
            if matches!(kind, PtrArgKind::Raw) {
                return None;
            }
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            let name = &pi.ident;
            Some(quote! { let #name = *#name; })
        })
        .collect();
    let block_stmts = &final_block.stmts;
    let function_stream: TokenStream2 = quote! {
        #[allow(non_snake_case)]
        #[allow(clippy::too_many_arguments)]
        #(#attrs)*
        #vis #host_sig {
            #(#ptr_unwraps)*
            #(#block_stmts)*
        }
    };

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

    // Fusion capability markers (metadata only — probe logic is the blanket in teeny-triton).
    let block_size_field = const_params
        .iter()
        .zip(const_field_idents.iter())
        .find(|(cp, _)| cp.ident == "BLOCK_SIZE")
        .map(|(_, field)| field.clone());

    let block_sized_impl = if let Some(field) = &block_size_field {
        quote! {
            impl #struct_generics_def ::teeny_triton::BlockSized
                for #struct_ident #struct_generics_use
            {
                fn block_size(&self) -> i32 {
                    self.#field
                }
            }
        }
    } else {
        quote! {}
    };

    let last_arg_is_n_elements = fn_inputs.last().is_some_and(|pt| {
        let name_ok = match &*pt.pat {
            Pat::Ident(pi) => pi.ident == "n_elements",
            _ => false,
        };
        let ty_ok = matches!(&*pt.ty, Type::Path(tp) if tp.path.is_ident("i32"));
        name_ok && ty_ok
    });

    let n_elements_tiled_impl = if block_size_field.is_some() && last_arg_is_n_elements {
        quote! {
            impl #struct_generics_def ::teeny_triton::NElementsTiled
                for #struct_ident #struct_generics_use
            {
            }
        }
    } else {
        quote! {}
    };

    // Inherent probe method so Dispatch (and fusion) can call it on every
    // kernel struct. Logic stays on the PointwiseFuseProbeExt blanket.
    let pointwise_probe_body = if block_size_field.is_some() && last_arg_is_n_elements {
        quote! {
            <Self as ::teeny_triton::PointwiseFuseProbeExt>::pointwise_fuse_probe(self)
        }
    } else {
        quote! { ::core::option::Option::None }
    };

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

            /// Pointer-parameter layout from this kernel's marked parameters.
            pub const fn kernel_io() -> ::teeny_triton::KernelIo {
                ::teeny_triton::KernelIo {
                    roles: &[ #(#ptr_roles),* ],
                }
            }

            /// Pointwise-fuse probe. Delegates to the
            /// [`::teeny_triton::PointwiseFuseProbeExt`] blanket when this
            /// kernel is `n_elements`-tiled unary elementwise; otherwise `None`.
            pub fn pointwise_fuse_probe(&self) -> ::core::option::Option<::teeny_triton::PointwiseFuseProbe> {
                #pointwise_probe_body
            }

            /// Splice-ready per-element compute for reduction-terminated
            /// fusion (teenygrad-3w0.9). See [`::teeny_triton::FusionCore`].
            #fusion_core_body

            #tile_spec_method

            #grid_spec_method
        }

        // Thin ABI metadata for fusion probing. Probe *logic* lives on
        // `PointwiseFuseProbeExt`'s blanket impl in `teeny-triton`, not here.
        impl #struct_generics_def ::teeny_triton::KernelIoLayout
            for #struct_ident #struct_generics_use
        {
            fn kernel_io() -> ::teeny_triton::KernelIo {
                ::teeny_triton::KernelIo {
                    roles: &[ #(#ptr_roles),* ],
                }
            }
        }

        #block_sized_impl
        #n_elements_tiled_impl

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
    //    via `#[tiled_kernel(dtypes = [..])]` and/or `#[tiled_kernel(backward = ..)]`.
    //    Maps a runtime `DtypeRepr` to the monomorphized kernel struct (and its
    //    paired backward, if declared), returning a crate-agnostic `KernelInstance`.
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
                    "cannot infer supported dtypes: a `#[tiled_kernel]` that opts into dispatch \
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
                        let __probe_bs = __f.pointwise_fuse_probe().map(|p| p.block_size);
                        let __body = __f.kernel_source.clone();
                        teeny_core::model::KernelInstance {
                            name: __f.name.to_string(),
                            source: __f.source.clone(),
                            kernel_body: __body,
                            runtime_op: ::std::sync::Arc::new(__f),
                            pointwise_fuse_block_size: __probe_bs,
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
