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

//! The `#[tiled_kernel]` macro: [`super::kernel::kernel`] plus the opinionated
//! tile-shape DSL (teenygrad-3w0) — `#[tile(...)]` on pointer parameters,
//! `#[tile_loop(...)]` and `#[tile_pid_swizzle(...)]` on the fn itself. Beyond
//! what the plain `#[kernel]` macro generates, this additionally produces:
//! auto-generated single-axis load preludes, GEMM's swizzled `pid` decode,
//! [`KernelTileSpec`]/[`TileSpecLayout`] for the cost model, and splice-ready
//! [`FusionCore`] bodies for reduction-terminated fusion. See
//! `kernels/teeny-triton/src/tile.rs` for the consumer side and
//! [`super::common`] for the parsing/codegen helpers shared with the plain
//! `#[kernel]` macro.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, FnArg, GenericParam, Ident, ItemFn, MetaNameValue, Pat, PatType, Token, Type,
    TypeParamBound, parse::Parser, parse_macro_input, punctuated::Punctuated,
};

use super::common::{
    PtrArgKind, all_dtypes_for_bound, classify_pointer_arg, dtype_ident_to_repr,
    extract_pointer_inner, parse_kernel_attrs, pat_to_str, simple_type_ident, to_pascal_case,
    unwrap_pointer_marker,
};

// ── Tile DSL parsing ────────────────────────────────────────────────────────

/// Remove `#[tile(...)]` attributes from every parameter of `sig`.
///
/// `#[tile(...)]` is host-only metadata consumed by this macro; like In/Out
/// pointer markers, it must never reach the final emitted Rust source (rustc
/// would reject an unrecognised parameter attribute).
fn strip_tile_attrs(sig: &mut syn::Signature) {
    for input in sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = input {
            pt.attrs.retain(|a| !a.path().is_ident("tile"));
        }
    }
}

/// Parsed `#[tile(block = BLOCK_SIZE, extent = n_elements)]` on one pointer param.
///
/// Phase 1a only: a single tile axis, naming a `const` generic (`block`) and
/// an `i32` parameter (`extent`) already present on the kernel's signature.
/// One entry per tile axis, in tensor-axis order — `block`/`extent` may each
/// be a single identifier (phase 1a: `block = BLOCK_SIZE`, one implicit axis)
/// or a list (phase 1b: `block = [BLOCK_M, BLOCK_K]`, one axis per element).
/// `reduction` optionally names which of *this tensor's own* axes (by index
/// into `block`/`extent`) is accumulated over in a loop, e.g. GEMM's K axis.
/// See `kernels/teeny-triton/src/tile.rs` / teenygrad-3w0.
#[derive(Debug)]
struct TileAttrArgs {
    block: Vec<Ident>,
    extent: Vec<Ident>,
    reduction: Option<usize>,
    /// Opt out of the single-axis auto-load-prelude codegen (default `true`)
    /// even though this tag alone would otherwise be eligible. Needed for
    /// kernels whose real `pid` decoding is more than `arange(BLOCK)+pid*BLOCK`
    /// (e.g. `conv2d_forward`'s multi-dim `pid % num_ow_tiles` split) — firing
    /// the simple prelude there would inject dead, wrong-shaped code rather
    /// than anything actually incorrect at runtime, but it's still misleading
    /// generated source, so kernel authors must explicitly say the simple
    /// shape doesn't apply.
    prelude: bool,
    /// `(stride, pad, kernel)` const-generic names for a strided/padded
    /// sliding-window axis (teenygrad-3w0.5), e.g. conv's `x_ptr`. Only
    /// valid on a single-axis (`block`/`extent` len == 1) tag — `stride`/
    /// `pad`/`kernel` must all be given together or not at all.
    window: Option<(Ident, Ident, Ident)>,
    /// Names of `{NAME}: i32` params giving this tensor's other real, but
    /// untiled, dimensions (teenygrad-3w0.8), e.g. conv's `y_ptr` declaring
    /// `untiled = [_B, C_OUT, OH]` alongside its tiled `OW` axis.
    untiled: Vec<Ident>,
}

/// Parse a `#[tile(...)]` value that must be a single bare identifier
/// (unlike `block`/`extent`, which also accept a list).
fn parse_single_ident(value: &Expr) -> Result<Ident, syn::Error> {
    let Expr::Path(p) = value else {
        return Err(syn::Error::new_spanned(
            value,
            "`#[tile(...)]` value must be a single identifier",
        ));
    };
    p.path.get_ident().cloned().ok_or_else(|| {
        syn::Error::new_spanned(value, "`#[tile(...)]` value must be a single identifier")
    })
}

/// Parse a `#[tile(...)]` value as either a bare identifier or a list of them.
fn parse_ident_or_list(value: &Expr) -> Result<Vec<Ident>, syn::Error> {
    match value {
        Expr::Path(p) => {
            let id = p.path.get_ident().ok_or_else(|| {
                syn::Error::new_spanned(value, "`#[tile(...)]` values must be a single identifier")
            })?;
            Ok(vec![id.clone()])
        }
        Expr::Array(arr) => arr
            .elems
            .iter()
            .map(|elem| {
                let Expr::Path(p) = elem else {
                    return Err(syn::Error::new_spanned(
                        elem,
                        "`#[tile(...)]` list elements must be bare identifiers",
                    ));
                };
                p.path.get_ident().cloned().ok_or_else(|| {
                    syn::Error::new_spanned(
                        elem,
                        "`#[tile(...)]` list elements must be a single identifier",
                    )
                })
            })
            .collect(),
        _ => Err(syn::Error::new_spanned(
            value,
            "`#[tile(...)]` values must be bare identifiers, e.g. `block = BLOCK_SIZE` or \
             `block = [BLOCK_M, BLOCK_K]`",
        )),
    }
}

/// Parse the `#[tile(...)]` attribute on one parameter, if present.
fn parse_tile_attr(pt: &PatType) -> Result<Option<TileAttrArgs>, syn::Error> {
    let Some(attr) = pt.attrs.iter().find(|a| a.path().is_ident("tile")) else {
        return Ok(None);
    };
    let meta_list = attr.meta.require_list()?;
    let parsed = Punctuated::<MetaNameValue, Token![,]>::parse_terminated
        .parse2(meta_list.tokens.clone())?;
    let mut block = None;
    let mut extent = None;
    let mut reduction = None;
    let mut prelude = true;
    let mut stride = None;
    let mut pad = None;
    let mut kernel = None;
    let mut untiled = Vec::new();
    for nv in parsed {
        let key = nv
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        match key.as_str() {
            "block" => block = Some(parse_ident_or_list(&nv.value)?),
            "extent" => extent = Some(parse_ident_or_list(&nv.value)?),
            "reduction" => {
                let Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Int(lit),
                    ..
                }) = &nv.value
                else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`#[tile(reduction = ...)]` must be an integer axis index",
                    ));
                };
                reduction = Some(lit.base10_parse::<usize>()?);
            }
            "prelude" => {
                let Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Bool(lit),
                    ..
                }) = &nv.value
                else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`#[tile(prelude = ...)]` must be `true` or `false`",
                    ));
                };
                prelude = lit.value;
            }
            "stride" => stride = Some(parse_single_ident(&nv.value)?),
            "pad" => pad = Some(parse_single_ident(&nv.value)?),
            "kernel" => kernel = Some(parse_single_ident(&nv.value)?),
            "untiled" => untiled = parse_ident_or_list(&nv.value)?,
            other => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    format!(
                        "unknown `#[tile(...)]` argument `{other}` (expected `block`, `extent`, \
                         `reduction`, `prelude`, `stride`, `pad`, `kernel`, or `untiled`)"
                    ),
                ));
            }
        }
    }
    let block = block.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile(...)]` requires `block = BLOCK_CONST`")
    })?;
    let extent = extent.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile(...)]` requires `extent = param_name`")
    })?;
    if block.len() != extent.len() {
        return Err(syn::Error::new_spanned(
            attr,
            format!(
                "`#[tile(...)]` has {} `block` entries but {} `extent` entries — they must \
                 name one axis each, in the same order",
                block.len(),
                extent.len()
            ),
        ));
    }
    if let Some(r) = reduction
        && r >= block.len()
    {
        return Err(syn::Error::new_spanned(
            attr,
            format!(
                "`#[tile(reduction = {r})]` is out of range — this tensor only has {} tile \
                 axes (0..{})",
                block.len(),
                block.len()
            ),
        ));
    }
    let window = match (stride, pad, kernel) {
        (Some(s), Some(p), Some(k)) => {
            if block.len() != 1 {
                return Err(syn::Error::new_spanned(
                    attr,
                    "`#[tile(stride = .., pad = .., kernel = ..)]` (a sliding-window axis) is \
                     only supported on a single-axis tag",
                ));
            }
            Some((s, p, k))
        }
        (None, None, None) => None,
        _ => {
            return Err(syn::Error::new_spanned(
                attr,
                "`#[tile(...)]`'s `stride`, `pad`, and `kernel` must all be given together, \
                 or not at all",
            ));
        }
    };
    Ok(Some(TileAttrArgs {
        block,
        extent,
        reduction,
        prelude,
        window,
        untiled,
    }))
}

/// Parsed `#[tile_loop(carry = [..], shape = [..], trip_count = ..)]` — a
/// function-level (not per-parameter) attribute describing a kernel's
/// loop-carried accumulator state (teenygrad-3w0.7), e.g. flash-attn's
/// online-softmax `acc`/`m_i`/`l_i`.
#[derive(Debug)]
struct TileLoopAttrArgs {
    carries: Vec<Ident>,
    /// Shape (per const-generic dimension) shared by every carry in
    /// `carries` — this codebase's loop-carried kernels only ever have
    /// same-shaped carries (Triton requires every `scf.for` iter-arg to
    /// agree in shape anyway), so one shared `shape` is sufficient rather
    /// than one per carry.
    shape: Vec<Ident>,
    trip_count: Ident,
}

/// Parse the `#[tile_loop(...)]` attribute on a kernel fn, if present.
fn parse_tile_loop_attr(attrs: &[syn::Attribute]) -> Result<Option<TileLoopAttrArgs>, syn::Error> {
    let Some(attr) = attrs.iter().find(|a| a.path().is_ident("tile_loop")) else {
        return Ok(None);
    };
    let meta_list = attr.meta.require_list()?;
    let parsed = Punctuated::<MetaNameValue, Token![,]>::parse_terminated
        .parse2(meta_list.tokens.clone())?;
    let mut carries = None;
    let mut shape = None;
    let mut trip_count = None;
    for nv in parsed {
        let key = nv
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        match key.as_str() {
            "carry" => carries = Some(parse_ident_or_list(&nv.value)?),
            "shape" => shape = Some(parse_ident_or_list(&nv.value)?),
            "trip_count" => trip_count = Some(parse_single_ident(&nv.value)?),
            other => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    format!(
                        "unknown `#[tile_loop(...)]` argument `{other}` (expected `carry`, `shape`, or `trip_count`)"
                    ),
                ));
            }
        }
    }
    let carries = carries.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_loop(...)]` requires `carry = [..]`")
    })?;
    let shape = shape.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_loop(...)]` requires `shape = [..]`")
    })?;
    let trip_count = trip_count.ok_or_else(|| {
        syn::Error::new_spanned(
            attr,
            "`#[tile_loop(...)]` requires `trip_count = param_name`",
        )
    })?;
    Ok(Some(TileLoopAttrArgs {
        carries,
        shape,
        trip_count,
    }))
}

/// Parsed `#[tile_pid_swizzle(block_m = .., block_n = .., m = .., n = .., group = ..)]`
/// — a function-level (not per-parameter) attribute generating GEMM's
/// L2-locality-swizzled `pid` decode (teenygrad-3w0.6), e.g.
/// `matmul_forward`'s `pid_m`/`pid_n` derivation. Purely a function of
/// scalar names already in the kernel signature — it never touches any
/// `#[tile(...)]`-tagged pointer param directly, so (unlike a per-parameter
/// tile axis) it needs no cross-tensor axis correspondence beyond what
/// `extent_param` name-sharing already gives `KernelTileSpec` today.
#[derive(Debug)]
struct TilePidSwizzleAttrArgs {
    block_m: Ident,
    block_n: Ident,
    m: Ident,
    n: Ident,
    group: Ident,
}

/// Parse the `#[tile_pid_swizzle(...)]` attribute on a kernel fn, if present.
fn parse_tile_pid_swizzle_attr(
    attrs: &[syn::Attribute],
) -> Result<Option<TilePidSwizzleAttrArgs>, syn::Error> {
    let Some(attr) = attrs.iter().find(|a| a.path().is_ident("tile_pid_swizzle")) else {
        return Ok(None);
    };
    let meta_list = attr.meta.require_list()?;
    let parsed = Punctuated::<MetaNameValue, Token![,]>::parse_terminated
        .parse2(meta_list.tokens.clone())?;
    let mut block_m = None;
    let mut block_n = None;
    let mut m = None;
    let mut n = None;
    let mut group = None;
    for nv in parsed {
        let key = nv
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        match key.as_str() {
            "block_m" => block_m = Some(parse_single_ident(&nv.value)?),
            "block_n" => block_n = Some(parse_single_ident(&nv.value)?),
            "m" => m = Some(parse_single_ident(&nv.value)?),
            "n" => n = Some(parse_single_ident(&nv.value)?),
            "group" => group = Some(parse_single_ident(&nv.value)?),
            other => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    format!(
                        "unknown `#[tile_pid_swizzle(...)]` argument `{other}` (expected \
                         `block_m`, `block_n`, `m`, `n`, or `group`)"
                    ),
                ));
            }
        }
    }
    let block_m = block_m.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_pid_swizzle(...)]` requires `block_m = ..`")
    })?;
    let block_n = block_n.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_pid_swizzle(...)]` requires `block_n = ..`")
    })?;
    let m = m.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_pid_swizzle(...)]` requires `m = ..`")
    })?;
    let n = n.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_pid_swizzle(...)]` requires `n = ..`")
    })?;
    let group = group.ok_or_else(|| {
        syn::Error::new_spanned(attr, "`#[tile_pid_swizzle(...)]` requires `group = ..`")
    })?;
    Ok(Some(TilePidSwizzleAttrArgs {
        block_m,
        block_n,
        m,
        n,
        group,
    }))
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
    let tile_loop_attr = match parse_tile_loop_attr(&input.attrs) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let tile_pid_swizzle_attr = match parse_tile_pid_swizzle_attr(&input.attrs) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    // `#[tile_loop(...)]`/`#[tile_pid_swizzle(...)]` are host-only metadata
    // (teenygrad-3w0.7/.6), like `#[tile(...)]` on parameters — strip them
    // before any re-emission so neither reaches the device-side source or
    // the generated host fn.
    let attrs: Vec<&syn::Attribute> = input
        .attrs
        .iter()
        .filter(|a| !a.path().is_ident("tile_loop") && !a.path().is_ident("tile_pid_swizzle"))
        .collect();
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

    // Parse and validate `#[tile(...)]` on each parameter (`In`/`Out`/`InOut`
    // pointers, each `block` entry naming a const generic, each `extent` entry
    // naming an `i32` arg, `reduction` — if given — indexing one of this
    // tensor's own tile axes). Metadata only — see
    // `kernels/teeny-triton/src/tile.rs` / teenygrad-3w0.
    let tile_attrs: Vec<Option<TileAttrArgs>> = {
        let mut out = Vec::with_capacity(fn_inputs.len());
        for pt in &fn_inputs {
            let parsed = match parse_tile_attr(pt) {
                Ok(p) => p,
                Err(e) => return e.to_compile_error().into(),
            };
            if let Some(args) = &parsed {
                let kind = classify_pointer_arg(&pt.ty, &hw_ident).map(|(k, _)| k);
                if !matches!(
                    kind,
                    Some(PtrArgKind::In) | Some(PtrArgKind::Out) | Some(PtrArgKind::InOut)
                ) {
                    let name = pat_to_str(&pt.pat);
                    return syn::Error::new_spanned(
                        &pt.ty,
                        format!(
                            "`#[tile(...)]` on `{name}` requires an `In`/`Out`/`InOut` parameter"
                        ),
                    )
                    .to_compile_error()
                    .into();
                }
                for block_ident in &args.block {
                    if !const_params.iter().any(|cp| cp.ident == *block_ident) {
                        return syn::Error::new_spanned(
                            block_ident,
                            format!(
                                "`#[tile(block = {block_ident})]` does not name a `const` generic on this kernel"
                            ),
                        )
                        .to_compile_error()
                        .into();
                    }
                }
                for extent_ident in &args.extent {
                    let extent_is_i32_param = fn_inputs.iter().any(|other| {
                        let name_ok =
                            matches!(&*other.pat, Pat::Ident(pi) if pi.ident == *extent_ident);
                        let ty_ok = matches!(&*other.ty, Type::Path(tp) if tp.path.is_ident("i32"));
                        name_ok && ty_ok
                    });
                    if !extent_is_i32_param {
                        return syn::Error::new_spanned(
                            extent_ident,
                            format!(
                                "`#[tile(extent = {extent_ident})]` does not name an `i32` parameter on this kernel"
                            ),
                        )
                        .to_compile_error()
                        .into();
                    }
                }
                if let Some((stride_ident, pad_ident, kernel_ident)) = &args.window {
                    for (key, ident) in [
                        ("stride", stride_ident),
                        ("pad", pad_ident),
                        ("kernel", kernel_ident),
                    ] {
                        if !const_params.iter().any(|cp| cp.ident == *ident) {
                            return syn::Error::new_spanned(
                                ident,
                                format!(
                                    "`#[tile({key} = {ident})]` does not name a `const` generic on this kernel"
                                ),
                            )
                            .to_compile_error()
                            .into();
                        }
                    }
                }
                for untiled_ident in &args.untiled {
                    let untiled_is_i32_param = fn_inputs.iter().any(|other| {
                        let name_ok =
                            matches!(&*other.pat, Pat::Ident(pi) if pi.ident == *untiled_ident);
                        let ty_ok = matches!(&*other.ty, Type::Path(tp) if tp.path.is_ident("i32"));
                        name_ok && ty_ok
                    });
                    if !untiled_is_i32_param {
                        return syn::Error::new_spanned(
                            untiled_ident,
                            format!(
                                "`#[tile(untiled = {untiled_ident})]` does not name an `i32` parameter on this kernel"
                            ),
                        )
                        .to_compile_error()
                        .into();
                    }
                }
            }
            out.push(parsed);
        }
        out
    };
    // Validate `#[tile_loop(...)]`, if present: `shape` idents must each
    // name a const generic (same rule as `#[tile(block = ...)]`), and
    // `trip_count` must name an `i32` parameter (same rule as `extent`).
    if let Some(args) = &tile_loop_attr {
        for shape_ident in &args.shape {
            if !const_params.iter().any(|cp| cp.ident == *shape_ident) {
                return syn::Error::new_spanned(
                    shape_ident,
                    format!(
                        "`#[tile_loop(shape = {shape_ident})]` does not name a `const` generic on this kernel"
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
        let trip_count_is_i32_param = fn_inputs.iter().any(|other| {
            let name_ok = matches!(&*other.pat, Pat::Ident(pi) if pi.ident == args.trip_count);
            let ty_ok = matches!(&*other.ty, Type::Path(tp) if tp.path.is_ident("i32"));
            name_ok && ty_ok
        });
        if !trip_count_is_i32_param {
            return syn::Error::new_spanned(
                &args.trip_count,
                format!(
                    "`#[tile_loop(trip_count = {})]` does not name an `i32` parameter on this kernel",
                    args.trip_count
                ),
            )
            .to_compile_error()
            .into();
        }
    }
    // Validate `#[tile_pid_swizzle(...)]`, if present: `block_m`/`block_n`/
    // `group` must each name a const generic (same rule as
    // `#[tile(block = ...)]`), and `m`/`n` must each name an `i32`
    // parameter (same rule as `extent`/`trip_count`).
    if let Some(args) = &tile_pid_swizzle_attr {
        for const_ident in [&args.block_m, &args.block_n, &args.group] {
            if !const_params.iter().any(|cp| cp.ident == *const_ident) {
                return syn::Error::new_spanned(
                    const_ident,
                    format!(
                        "`#[tile_pid_swizzle(...)]`'s `{const_ident}` does not name a `const` generic on this kernel"
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
        for param_ident in [&args.m, &args.n] {
            let is_i32_param = fn_inputs.iter().any(|other| {
                let name_ok = matches!(&*other.pat, Pat::Ident(pi) if pi.ident == *param_ident);
                let ty_ok = matches!(&*other.ty, Type::Path(tp) if tp.path.is_ident("i32"));
                name_ok && ty_ok
            });
            if !is_i32_param {
                return syn::Error::new_spanned(
                    param_ident,
                    format!(
                        "`#[tile_pid_swizzle(...)]`'s `{param_ident}` does not name an `i32` parameter on this kernel"
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
    }
    let has_tile_spec = tile_attrs.iter().any(Option::is_some) || tile_loop_attr.is_some();

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

    // Auto-generated load prelude (phase 1a stretch goal — teenygrad-3w0.1):
    // when every `#[tile(...)]`-tagged pointer is single-axis and they all
    // share one `(block, extent)` pair, synthesize the `pid`/`block_start`/
    // `offsets`/`in_bounds`/load boilerplate every flat elementwise kernel
    // repeats by hand, and prepend it to the body. `T::store(...)` stays the
    // kernel author's explicit last statement (reusing the auto-bound
    // `offsets`/`in_bounds`), so this only removes load-side duplication, not
    // store-side flexibility. Multi-axis tags (phase 1b — teenygrad-3w0.2,
    // e.g. GEMM's per-tensor `[BLOCK_M, BLOCK_K]`) are metadata-only: the
    // pid-decode/indexing they'd need (swizzle, reduction loops, strided
    // sliding windows) isn't a safe fit for this single generated prelude, so
    // they're deliberately excluded here rather than guessed at.
    let prelude_group: Option<(&Ident, &Ident)> = {
        let mut single_axis = tile_attrs
            .iter()
            .flatten()
            .filter(|a| a.prelude && a.block.len() == 1 && a.extent.len() == 1);
        single_axis.next().and_then(|first| {
            if single_axis.all(|a| a.block[0] == first.block[0] && a.extent[0] == first.extent[0]) {
                Some((&first.block[0], &first.extent[0]))
            } else {
                None
            }
        })
    };
    let prelude_stmts: Vec<syn::Stmt> = match prelude_group {
        Some((block_const, extent_param)) => {
            let mut stmts: Vec<syn::Stmt> = syn::parse2::<syn::Block>(quote! {{
                let pid = #hw_ident::program_id(Axis::X);
                let block_start = pid * #block_const;
                let offsets = #hw_ident::arange(0, #block_const) + block_start;
                let in_bounds = offsets.lt(#extent_param);
            }})
            .expect("generated tile prelude is valid Rust")
            .stmts;
            for (pt, tile_attr) in fn_inputs.iter().zip(tile_attrs.iter()) {
                if tile_attr.is_none() {
                    continue;
                }
                let Some((PtrArgKind::In, _)) = classify_pointer_arg(&pt.ty, &hw_ident) else {
                    continue;
                };
                let Pat::Ident(pi) = &*pt.pat else { continue };
                let ptr_ident = &pi.ident;
                let ptr_name = ptr_ident.to_string();
                let base_name = ptr_name.strip_suffix("_ptr").unwrap_or(&ptr_name);
                let base_ident = Ident::new(base_name, ptr_ident.span());
                let load_stmt: syn::Stmt = syn::parse2(quote! {
                    let #base_ident = #hw_ident::load(
                        #ptr_ident.add_offsets(offsets),
                        Some(in_bounds),
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
                })
                .expect("generated tile load statement is valid Rust");
                stmts.push(load_stmt);
            }
            stmts
        }
        None => Vec::new(),
    };
    // GEMM-swizzle pid-decode prelude (teenygrad-3w0.6): generates the exact
    // statement sequence `matmul_forward` used to hand-write (same local
    // names throughout, so the rest of the body can reference `pid_m`/
    // `pid_n` unchanged and the generated source stays byte-identical to
    // the hand-written original). Additive with the single-axis
    // `prelude_stmts` above -- a kernel using `#[tile_pid_swizzle(...)]`
    // has multi-axis `#[tile(...)]` tags in every case seen so far, so
    // `prelude_group` is `None` and `prelude_stmts` is empty in practice,
    // but nothing here assumes that.
    let swizzle_stmts: Vec<syn::Stmt> = match &tile_pid_swizzle_attr {
        Some(args) => {
            let block_m = &args.block_m;
            let block_n = &args.block_n;
            let m = &args.m;
            let n = &args.n;
            let group = &args.group;
            syn::parse2::<syn::Block>(quote! {{
                let pid = #hw_ident::program_id(Axis::X);
                let num_pid_m = #hw_ident::cdiv(#m, #block_m);
                let num_pid_n = #hw_ident::cdiv(#n, #block_n);
                let num_pid_in_group = #group * num_pid_n;
                let group_id = pid / num_pid_in_group;
                let first_pid_m = group_id * #group;
                let remaining_m = num_pid_m - first_pid_m;
                let group_size_m = if remaining_m < #group {
                    remaining_m
                } else {
                    #group
                };
                let pid_in_group = pid % num_pid_in_group;
                let pid_m = first_pid_m + (pid_in_group % group_size_m);
                let pid_n = pid_in_group / group_size_m;
            }})
            .expect("generated swizzle prelude is valid Rust")
            .stmts
        }
        None => Vec::new(),
    };
    let final_stmts: Vec<syn::Stmt> = swizzle_stmts
        .into_iter()
        .chain(prelude_stmts)
        .chain(input.block.stmts.iter().cloned())
        .collect();
    let final_block = syn::Block {
        brace_token: input.block.brace_token,
        stmts: final_stmts,
    };

    // Splice-ready "core" for reduction-terminated fusion (teenygrad-3w0.9):
    // available only for single-input, single-axis prelude kernels (exactly
    // one `#[tile(...)]`-tagged `In`, sharing `prelude_group`'s block/
    // extent) whose hand-written body's last statement is a plain
    // `#hw_ident::store(...)` call -- the shape every such kernel in this
    // tree today uses. Extraction happens here (proc-macro time, full `syn`
    // AST) rather than as a runtime string operation on generated source:
    // the "last computed identifier" (this kernel's `output_ident`) varies
    // per kernel and isn't reliably recoverable from text alone.
    let single_input_ident: Option<String> = prelude_group.and_then(|_| {
        let mut in_tile_idents =
            fn_inputs
                .iter()
                .zip(tile_attrs.iter())
                .filter_map(|(pt, attr)| {
                    attr.as_ref()?;
                    let Some((PtrArgKind::In, _)) = classify_pointer_arg(&pt.ty, &hw_ident) else {
                        return None;
                    };
                    let Pat::Ident(pi) = &*pt.pat else {
                        return None;
                    };
                    // Matches the prelude loop's own load-ident derivation
                    // (`x_ptr` -> `x`) so `input_ident` names the same local
                    // the auto-prelude actually binds.
                    let ptr_name = pi.ident.to_string();
                    let base_name = ptr_name
                        .strip_suffix("_ptr")
                        .unwrap_or(&ptr_name)
                        .to_string();
                    Some(base_name)
                });
        let first = in_tile_idents.next()?;
        if in_tile_idents.next().is_none() {
            Some(first)
        } else {
            None
        }
    });

    let trailing_store_core: Option<&[syn::Stmt]> = single_input_ident.as_ref().and_then(|_| {
        let stmts = &input.block.stmts;
        let last = stmts.last()?;
        let syn::Stmt::Expr(Expr::Call(call), Some(_)) = last else {
            return None;
        };
        let Expr::Path(path) = &*call.func else {
            return None;
        };
        let segs = &path.path.segments;
        if segs.len() == 2 && segs[0].ident == hw_ident && segs[1].ident == "store" {
            Some(&stmts[..stmts.len() - 1])
        } else {
            None
        }
    });

    let fusion_core_none = quote! {
        pub fn fusion_core() -> ::core::option::Option<::teeny_triton::FusionCore> {
            ::core::option::Option::None
        }
    };
    let fusion_core_body: TokenStream2 = match (single_input_ident, trailing_store_core) {
        (Some(input_ident), Some(core_stmts)) if !core_stmts.is_empty() => {
            let output_ident_str = match core_stmts.last() {
                Some(syn::Stmt::Local(local)) => match &local.pat {
                    Pat::Ident(pi) => Some(pi.ident.to_string()),
                    _ => None,
                },
                _ => None,
            };
            match output_ident_str {
                Some(output_ident_str) => {
                    let input_ident_str = input_ident;
                    let body_source = quote!(#(#core_stmts)*).to_string();
                    let extra_param_tokens: Vec<TokenStream2> = fn_inputs
                        .iter()
                        .filter_map(|pt| {
                            if classify_pointer_arg(&pt.ty, &hw_ident).is_some() {
                                return None;
                            }
                            let Pat::Ident(pi) = &*pt.pat else {
                                return None;
                            };
                            if let Some((_, extent_param)) = prelude_group
                                && pi.ident == *extent_param
                            {
                                return None;
                            }
                            let name_str = pi.ident.to_string();
                            let ty = &pt.ty;
                            let ty_str = quote!(#ty).to_string();
                            Some(quote! { (#name_str, #ty_str) })
                        })
                        .collect();
                    quote! {
                        pub fn fusion_core() -> ::core::option::Option<::teeny_triton::FusionCore> {
                            ::core::option::Option::Some(::teeny_triton::FusionCore {
                                input_ident: #input_ident_str,
                                output_ident: #output_ident_str,
                                body_source: #body_source,
                                extra_params: &[ #(#extra_param_tokens),* ],
                            })
                        }
                    }
                }
                None => fusion_core_none.clone(),
            }
        }
        _ => fusion_core_none,
    };

    // Device-side source: same body, but pointer markers stripped so MLIR sees
    // bare `T::Pointer<D>`. Host keeps the marked signature for KernelIo / API.
    let mut device_sig = sig.clone();
    for input in device_sig.inputs.iter_mut() {
        if let FnArg::Typed(pt) = input {
            *pt.ty = unwrap_pointer_marker(&pt.ty);
        }
    }
    strip_tile_attrs(&mut device_sig);
    let original_source_str = quote!(#vis #device_sig #final_block).to_string();

    // Host-side signature: same as the original, minus `#[tile(...)]` markers
    // (host-only metadata for `KernelTileSpec`, never valid on emitted Rust).
    let mut host_sig = sig.clone();
    strip_tile_attrs(&mut host_sig);

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

    // Build `TensorTileSpec`s for each `#[tile(...)]`-tagged input/output pointer.
    // Per-attribute validation above (block names a const generic, extent names
    // an i32 param) is the only gate `#[tile(...)]` needs — it does not require
    // `n_elements` to be the trailing argument the way `NElementsTiled` does,
    // since that's a separate, narrower contract for the fusion-probe blanket.
    let mut input_tile_specs: Vec<TokenStream2> = Vec::new();
    let mut output_tile_specs: Vec<TokenStream2> = Vec::new();
    for (pt, tile_attr) in fn_inputs.iter().zip(tile_attrs.iter()) {
        let Some(args) = tile_attr else { continue };
        let param_str = pat_to_str(&pt.pat);
        let rank = args.block.len();
        let window_tokens = match &args.window {
            Some((stride_ident, pad_ident, kernel_ident)) => {
                let stride_str = stride_ident.to_string();
                let pad_str = pad_ident.to_string();
                let kernel_str = kernel_ident.to_string();
                quote! {
                    ::core::option::Option::Some(::teeny_triton::tile::TileWindow {
                        stride_const: #stride_str,
                        pad_const: #pad_str,
                        kernel_size_const: #kernel_str,
                    })
                }
            }
            None => quote! { ::core::option::Option::None },
        };
        let axis_specs: Vec<TokenStream2> = args
            .block
            .iter()
            .zip(args.extent.iter())
            .enumerate()
            .map(|(dim, (block_ident, extent_ident))| {
                let block_str = block_ident.to_string();
                let extent_str = extent_ident.to_string();
                quote! {
                    ::teeny_triton::TileAxisBinding {
                        dim: #dim,
                        block_const: #block_str,
                        extent_param: #extent_str,
                        window: #window_tokens,
                    }
                }
            })
            .collect();
        let reduction_axis = match args.reduction {
            Some(r) => quote! { ::core::option::Option::Some(#r) },
            None => quote! { ::core::option::Option::None },
        };
        let untiled_strs: Vec<String> = args.untiled.iter().map(|u| u.to_string()).collect();
        let spec = quote! {
            ::teeny_triton::TensorTileSpec {
                param: #param_str,
                rank: #rank,
                axes: &[ #(#axis_specs),* ],
                reduction_axis: #reduction_axis,
                untiled_dims: &[ #(#untiled_strs),* ],
            }
        };
        let (kind, _) = classify_pointer_arg(&pt.ty, &hw_ident).expect("validated above");
        match kind {
            PtrArgKind::In => input_tile_specs.push(spec),
            PtrArgKind::Out => output_tile_specs.push(spec),
            // In-place: `PtrRole::InOut` is "read+write" by contract even when
            // a specific kernel body only writes it (e.g. GEMM's `c_ptr`), so
            // it conservatively belongs in both lists rather than guessing
            // from the body which axis relationship actually applies.
            PtrArgKind::InOut => {
                input_tile_specs.push(spec.clone());
                output_tile_specs.push(spec);
            }
            PtrArgKind::Raw => unreachable!("rejected above"),
        }
    }

    let loop_spec_tokens: TokenStream2 = match &tile_loop_attr {
        Some(args) => {
            let shape_strs: Vec<String> = args.shape.iter().map(|s| s.to_string()).collect();
            let carry_specs: Vec<TokenStream2> = args
                .carries
                .iter()
                .map(|carry_ident| {
                    let carry_str = carry_ident.to_string();
                    quote! {
                        ::teeny_triton::TileCarryBinding {
                            name: #carry_str,
                            shape_consts: &[ #(#shape_strs),* ],
                        }
                    }
                })
                .collect();
            let trip_count_str = args.trip_count.to_string();
            quote! {
                ::core::option::Option::Some(::teeny_triton::TileLoopSpec {
                    carries: &[ #(#carry_specs),* ],
                    trip_count_param: #trip_count_str,
                })
            }
        }
        None => quote! { ::core::option::Option::None },
    };

    let tile_spec_impl: TokenStream2 = if has_tile_spec {
        quote! {
            impl #struct_generics_def #struct_ident #struct_generics_use {
                /// Declared tile shape from this kernel's `#[tile(...)]`-annotated
                /// parameters. See `teenygrad-3w0`.
                pub const fn tile_spec() -> ::teeny_triton::KernelTileSpec {
                    ::teeny_triton::KernelTileSpec {
                        inputs: &[ #(#input_tile_specs),* ],
                        outputs: &[ #(#output_tile_specs),* ],
                        loop_spec: #loop_spec_tokens,
                    }
                }
            }

            impl #struct_generics_def ::teeny_triton::TileSpecLayout
                for #struct_ident #struct_generics_use
            {
                fn tile_spec() -> ::teeny_triton::KernelTileSpec {
                    ::teeny_triton::KernelTileSpec {
                        inputs: &[ #(#input_tile_specs),* ],
                        outputs: &[ #(#output_tile_specs),* ],
                        loop_spec: #loop_spec_tokens,
                    }
                }
            }
        }
    } else {
        quote! {}
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
    result.extend(TokenStream::from(tile_spec_impl));
    result.extend(TokenStream::from(dispatcher_stream));
    result
}

// ── Tests: `#[tile(...)]` attribute parsing (teenygrad-3w0.1) ─────────────────
//
// `parse_tile_attr`/`strip_tile_attrs` operate on `syn`/`proc_macro2` types
// (not `proc_macro::TokenStream`), so — unlike `tiled_kernel()` itself —
// they're directly callable from a normal unit test without an active macro
// invocation. This is cheaper and more precise than a `trybuild` UI test
// here: it asserts on the actual `syn::Error` message, not compiler-version-
// sensitive stderr output, and avoids adding a `teeny-triton`/`teeny-core`
// dev-dependency cycle back onto this crate purely to give UI fixtures
// something to `use`.
#[cfg(test)]
mod tests {
    use super::*;

    /// `PatType::parse` alone doesn't accept a leading outer attribute (that's
    /// normally attached by the surrounding `FnArg`/signature parser), so
    /// build a fixture through a full one-argument `fn` signature instead.
    fn pat_type(arg_tokens: TokenStream2) -> PatType {
        let sig: syn::Signature = syn::parse2(quote! { fn f(#arg_tokens) })
            .expect("valid single-argument Signature fixture");
        let FnArg::Typed(pt) = sig.inputs.into_iter().next().expect("one argument") else {
            panic!("expected a typed argument");
        };
        pt
    }

    #[test]
    fn parse_tile_attr_absent_returns_none() {
        let pt = pat_type(quote! { x_ptr: In<T::Pointer<D>> });
        assert!(parse_tile_attr(&pt).unwrap().is_none());
    }

    #[test]
    fn parse_tile_attr_valid_returns_block_and_extent() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_SIZE, extent = n_elements)]
            x_ptr: In<T::Pointer<D>>
        });
        let args = parse_tile_attr(&pt).unwrap().expect("attribute present");
        assert_eq!(args.block, vec!["BLOCK_SIZE"]);
        assert_eq!(args.extent, vec!["n_elements"]);
        assert_eq!(args.reduction, None);
        assert!(args.prelude, "prelude defaults to true");
    }

    #[test]
    fn parse_tile_attr_prelude_false_opts_out() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_OW, extent = OW, prelude = false)]
            y_ptr: Out<T::Pointer<D>>
        });
        let args = parse_tile_attr(&pt).unwrap().expect("attribute present");
        assert!(!args.prelude);
    }

    #[test]
    fn parse_tile_attr_window_parses_stride_pad_kernel() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_OW, extent = W, stride = STRIDE_W, pad = PAD_W, kernel = KW, prelude = false)]
            x_ptr: In<T::Pointer<D>>
        });
        let args = parse_tile_attr(&pt).unwrap().expect("attribute present");
        let (stride, pad, kernel) = args.window.expect("window present");
        assert_eq!(stride, "STRIDE_W");
        assert_eq!(pad, "PAD_W");
        assert_eq!(kernel, "KW");
    }

    #[test]
    fn parse_tile_attr_untiled_parses_dim_list() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_OW, extent = OW, untiled = [_B, C_OUT, OH])]
            y_ptr: Out<T::Pointer<D>>
        });
        let args = parse_tile_attr(&pt).unwrap().expect("attribute present");
        assert_eq!(args.untiled, vec!["_B", "C_OUT", "OH"]);
    }

    #[test]
    fn parse_tile_attr_untiled_defaults_to_empty() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_SIZE, extent = n_elements)]
            x_ptr: In<T::Pointer<D>>
        });
        let args = parse_tile_attr(&pt).unwrap().expect("attribute present");
        assert!(args.untiled.is_empty());
    }

    #[test]
    fn parse_tile_attr_window_requires_all_three_together() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_OW, extent = W, stride = STRIDE_W)]
            x_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(
            err.to_string().contains("must all be given together"),
            "{err}"
        );
    }

    #[test]
    fn parse_tile_attr_window_rejects_multi_axis() {
        let pt = pat_type(quote! {
            #[tile(block = [BLOCK_M, BLOCK_K], extent = [M, K], stride = S, pad = P, kernel = K2)]
            a_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(err.to_string().contains("single-axis tag"), "{err}");
    }

    #[test]
    fn parse_tile_attr_missing_block_is_an_error() {
        let pt = pat_type(quote! {
            #[tile(extent = n_elements)]
            x_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(err.to_string().contains("block = BLOCK_CONST"), "{err}");
    }

    #[test]
    fn parse_tile_attr_missing_extent_is_an_error() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_SIZE)]
            x_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(err.to_string().contains("extent = param_name"), "{err}");
    }

    #[test]
    fn parse_tile_attr_unknown_key_is_an_error() {
        let pt = pat_type(quote! {
            #[tile(block = BLOCK_SIZE, extent = n_elements, axis = X)]
            x_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(
            err.to_string().contains("unknown `#[tile(...)]` argument"),
            "{err}"
        );
        assert!(err.to_string().contains("axis"), "{err}");
    }

    #[test]
    fn parse_tile_attr_non_ident_value_is_an_error() {
        let pt = pat_type(quote! {
            #[tile(block = 1024, extent = n_elements)]
            x_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(err.to_string().contains("bare identifiers"), "{err}");
    }

    #[test]
    fn parse_tile_attr_multi_axis_with_reduction() {
        let pt = pat_type(quote! {
            #[tile(block = [BLOCK_M, BLOCK_K], extent = [M, K], reduction = 1)]
            a_ptr: In<T::Pointer<D>>
        });
        let args = parse_tile_attr(&pt).unwrap().expect("attribute present");
        assert_eq!(args.block, vec!["BLOCK_M", "BLOCK_K"]);
        assert_eq!(args.extent, vec!["M", "K"]);
        assert_eq!(args.reduction, Some(1));
    }

    #[test]
    fn parse_tile_attr_mismatched_block_extent_lengths_is_an_error() {
        let pt = pat_type(quote! {
            #[tile(block = [BLOCK_M, BLOCK_K], extent = [M])]
            a_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(
            err.to_string()
                .contains("2 `block` entries but 1 `extent` entries"),
            "{err}"
        );
    }

    #[test]
    fn parse_tile_attr_reduction_out_of_range_is_an_error() {
        let pt = pat_type(quote! {
            #[tile(block = [BLOCK_M, BLOCK_K], extent = [M, K], reduction = 2)]
            a_ptr: In<T::Pointer<D>>
        });
        let err = parse_tile_attr(&pt).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }

    #[test]
    fn strip_tile_attrs_removes_tile_but_keeps_other_attrs() {
        let mut sig: syn::Signature = syn::parse2(quote! {
            fn f(
                #[tile(block = BLOCK_SIZE, extent = n_elements)]
                #[allow(unused)]
                x_ptr: In<T::Pointer<D>>,
                n_elements: i32,
            )
        })
        .expect("valid Signature fixture");
        strip_tile_attrs(&mut sig);
        let FnArg::Typed(pt) = &sig.inputs[0] else {
            panic!("expected typed arg")
        };
        assert!(!pt.attrs.iter().any(|a| a.path().is_ident("tile")));
        assert!(pt.attrs.iter().any(|a| a.path().is_ident("allow")));
    }

    fn item_fn(tokens: TokenStream2) -> ItemFn {
        syn::parse2(tokens).expect("valid ItemFn fixture")
    }

    #[test]
    fn parse_tile_loop_attr_absent_returns_none() {
        let f = item_fn(quote! { fn f(n_ctx_k: i32) {} });
        assert!(parse_tile_loop_attr(&f.attrs).unwrap().is_none());
    }

    #[test]
    fn parse_tile_loop_attr_valid_parses_all_fields() {
        let f = item_fn(quote! {
            #[tile_loop(carry = [acc, m_i, l_i], shape = [HEAD_DIM], trip_count = n_ctx_k)]
            fn f(n_ctx_k: i32) {}
        });
        let args = parse_tile_loop_attr(&f.attrs)
            .unwrap()
            .expect("attribute present");
        assert_eq!(args.carries, vec!["acc", "m_i", "l_i"]);
        assert_eq!(args.shape, vec!["HEAD_DIM"]);
        assert_eq!(args.trip_count, "n_ctx_k");
    }

    #[test]
    fn parse_tile_loop_attr_missing_carry_is_an_error() {
        let f = item_fn(quote! {
            #[tile_loop(shape = [HEAD_DIM], trip_count = n_ctx_k)]
            fn f(n_ctx_k: i32) {}
        });
        let err = parse_tile_loop_attr(&f.attrs).unwrap_err();
        assert!(err.to_string().contains("carry = [..]"), "{err}");
    }

    #[test]
    fn parse_tile_pid_swizzle_attr_absent_returns_none() {
        let f = item_fn(quote! { fn f(m: i32, n: i32) {} });
        assert!(parse_tile_pid_swizzle_attr(&f.attrs).unwrap().is_none());
    }

    #[test]
    fn parse_tile_pid_swizzle_attr_valid_parses_all_fields() {
        let f = item_fn(quote! {
            #[tile_pid_swizzle(block_m = BLOCK_M, block_n = BLOCK_N, m = M, n = N, group = GROUP_M)]
            fn f(m: i32, n: i32) {}
        });
        let args = parse_tile_pid_swizzle_attr(&f.attrs)
            .unwrap()
            .expect("attribute present");
        assert_eq!(args.block_m, "BLOCK_M");
        assert_eq!(args.block_n, "BLOCK_N");
        assert_eq!(args.m, "M");
        assert_eq!(args.n, "N");
        assert_eq!(args.group, "GROUP_M");
    }

    #[test]
    fn parse_tile_pid_swizzle_attr_missing_group_is_an_error() {
        let f = item_fn(quote! {
            #[tile_pid_swizzle(block_m = BLOCK_M, block_n = BLOCK_N, m = M, n = N)]
            fn f(m: i32, n: i32) {}
        });
        let err = parse_tile_pid_swizzle_attr(&f.attrs).unwrap_err();
        assert!(err.to_string().contains("group = .."), "{err}");
    }

    #[test]
    fn parse_tile_pid_swizzle_attr_unknown_key_is_an_error() {
        let f = item_fn(quote! {
            #[tile_pid_swizzle(block_m = BLOCK_M, block_n = BLOCK_N, m = M, n = N, group = GROUP_M, axis = X)]
            fn f(m: i32, n: i32) {}
        });
        let err = parse_tile_pid_swizzle_attr(&f.attrs).unwrap_err();
        assert!(
            err.to_string()
                .contains("unknown `#[tile_pid_swizzle(...)]` argument"),
            "{err}"
        );
    }
}
