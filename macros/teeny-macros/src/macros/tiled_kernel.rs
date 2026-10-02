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
//! `84ca6eedf`'s mistake: it drives the auto-prelude's own block/extent
//! naming and this file's generated `tile_spec()` method, and it never
//! re-splices index arithmetic into the kernel author's own body.
//!
//! teenygrad-1nr.18.1 then generalized the prelude from that one flat
//! `arange(block)+pid*block` axis to N declared axes: a flat `program_id`
//! decoded innermost-first, row-major strides derived from the declared
//! extents, a `tile_<name>` index bound per axis, and broadcast subsets
//! (an input may declare fewer axes than the output), and .18.5 generalized
//! it again to several blocked axes at once.
//!
//! The attribute vocabulary has since caught up with most of the spec type.
//! `window(stride, pad, kernel, output)` emits a real `TileWindow`
//! (teenygrad-1nr.18.2), the bare `reduce` flag sets `reduction_axis`, and
//! `#[tile_loop]`/`#[tile_carry]` emit a `TileLoopSpec` (.18.3). Only
//! `divide_by` is still a hardcoded `None`, because GroupNorm's
//! `channels_per_group` is a runtime value while `tile_spec()` returns
//! `&'static` data -- teenygrad-1nr.15, and the reasoning is written out in
//! `groupnorm.rs`.
//!
//! Two function-level attributes sit beside those: `#[tile_loop]`/
//! `#[tile_carry]` above, and `#[tile_grid]`, which says what
//! `grid_spec()` cannot otherwise know -- the body's real `pid` decode order
//! when it differs from the output's dim order, which axes are grid axes at
//! all when a loop covers one, or `swizzled` when the decode is not a
//! mixed-radix decode of those axes and so has no `GridSpec` at all
//! (teenygrad-1tl.5, .1tl.8, .1tl.10). Note the *prelude* still generates no
//! windowed or looped index arithmetic: these attributes describe a
//! route-2 kernel's hand-written body, they do not write it.
//!
//! Optional and additive: a `Tile` parameter with no `#[tile(...)]` falls
//! back to the pre-existing hardcoded `BLOCK_SIZE`/`n_elements`
//! convention, unchanged, and no `tile_spec()` is generated for it. That
//! fallback is silent, which is why `teeny-kernels`'
//! `tests/test_tile_declarations.rs` parses the source to catch a forward
//! kernel that takes `Tile` params and declares nothing.
//!
//! **Converting a kernel: see `contributing/TiledKernelRecipe.md`** for the
//! recipe, the worked examples, the four attribute keys and what they
//! cannot express yet. See [`super::common`] for the parsing/codegen
//! helpers shared with the plain `#[kernel]` macro.

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
#[derive(Clone)]
struct TileAttrArgs {
    /// `Some(block_const)` when this axis is block-tiled (one CTA covers
    /// `block_const` elements); `None` for an untiled axis (one CTA per
    /// index). Required (`Some`) on a `Tile`-typed parameter -- untiled
    /// `Tile` axes aren't supported yet, see the auto-prelude's own
    /// requirements.
    block: Option<String>,
    /// The `{NAME}: i32` parameter this axis's extent is read from.
    extent: Ident,
    /// This axis's identity for [`::teeny_core::model::GridSpec`]
    /// matching (teenygrad-1nr.19), and the suffix of the index the
    /// prelude binds for it (`name = "C"` gives `tile_c`).
    ///
    /// Defaults to `extent`'s own spelling, so **omit it unless the two
    /// genuinely differ** -- `conv2d_forward`'s batch axis is `B` while
    /// its parameter is `_B`, which is the only case in the tree that
    /// needs it.
    name: Option<syn::LitStr>,
    /// Which real hardware grid dimension this axis reads from
    /// (teenygrad-1nr.19) -- `X`, `Y`, or `Z`; defaults to `X`.
    dim: Option<Ident>,
    /// `Some((stride, pad, kernel))` when this axis is read through a strided,
    /// padded sliding window, declared as
    /// `#[tile(extent = H, window(stride = STRIDE_H, pad = PAD_H, kernel = KH))]`
    /// (teenygrad-1nr.18.2).
    ///
    /// `stride`, `pad` and `kernel` are each the name of a `const {NAME}: i32`
    /// generic *or* a decimal integer literal; `output` always names a runtime
    /// parameter. An output tile of `block` elements along this axis reads a
    /// receptive field of `(block - 1) * stride + kernel` input elements --
    /// forward and exact. Padding shifts the window's origin, not its size, so
    /// it does not appear in that extent: an interior tile touches no padding
    /// at all.
    ///
    /// Literals exist for the pad family, which has no stride or kernel const
    /// to name: padding is a window of `stride = 1, kernel = 1` -- each output
    /// element reads exactly one input element, at an origin shifted by the
    /// pad -- giving `(block - 1) * 1 + 1 = block`, the tile's own width
    /// (teenygrad-1tl.5).
    window: Option<(String, Option<String>, String, Ident)>,
    /// `true` when this axis is the one the tensor is reduced over, declared
    /// as a bare `#[tile(extent = N, reduce)]` (teenygrad-1tl.8).
    ///
    /// A reduced axis cannot be tiled -- a row's mean needs the whole row --
    /// so this is what tells `TileGraph::propagate` which axis is *not*
    /// available for tiling, rather than leaving it to be inferred from the
    /// absence of a block.
    reduce: bool,
}

/// The `TileWindow` an axis declares, or `None` when it is read contiguously.
fn window_tokens(axis: &TileAttrArgs) -> TokenStream2 {
    match &axis.window {
        None => quote! { ::core::option::Option::None },
        Some((stride, pad, kernel, output)) => {
            let (s, k) = (stride.clone(), kernel.clone());
            let o = output.to_string();
            let p = match pad {
                Some(pad) => {
                    let p = pad.clone();
                    quote! { ::core::option::Option::Some(#p) }
                }
                None => quote! { ::core::option::Option::None },
            };
            quote! {
                ::core::option::Option::Some(::teeny_core::model::TileWindow {
                    output_extent_param: #o,
                    stride_const: #s,
                    pad_const: #p,
                    kernel_size_const: #k,
                })
            }
        }
    }
}

/// The index of the axis a tensor is reduced over, if one declared `reduce`.
///
/// Rejects more than one: `TensorTileSpec::reduction_axis` is a single index, and
/// a tensor reduced over two axes at once is not expressible (teenygrad-1tl.8).
fn reduction_axis_of(attrs: &[TileAttrArgs]) -> Result<Option<usize>, syn::Error> {
    let marked: Vec<usize> = attrs
        .iter()
        .enumerate()
        .filter(|(_, a)| a.reduce)
        .map(|(i, _)| i)
        .collect();
    match marked.as_slice() {
        [] => Ok(None),
        [i] => Ok(Some(*i)),
        _ => Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            format!(
                "{} axes declare `reduce`, but `reduction_axis` holds a single index; a tensor \
                 reduced over several axes at once is not expressible today",
                marked.len()
            ),
        )),
    }
}

/// Parse one `#[tile(...)]` attribute's contents.
fn parse_one_tile_attr(attr: &syn::Attribute) -> Result<TileAttrArgs, syn::Error> {
    let meta_list = attr.meta.require_list()?;
    // `Meta`, not `MetaNameValue`: an axis may carry bare flags (`reduce`)
    // alongside `key = value` pairs.
    let parsed =
        Punctuated::<syn::Meta, Token![,]>::parse_terminated.parse2(meta_list.tokens.clone())?;
    let mut block = None;
    let mut extent = None;
    let mut name = None;
    let mut dim = None;
    let mut reduce = false;
    let mut window = None;
    let mut nvs: Vec<MetaNameValue> = Vec::new();
    for meta in parsed {
        match meta {
            syn::Meta::Path(path) if path.is_ident("reduce") => reduce = true,
            syn::Meta::Path(path) => {
                return Err(syn::Error::new_spanned(
                    &path,
                    "unknown `#[tile(...)]` flag (expected `reduce`)",
                ));
            }
            syn::Meta::NameValue(nv) => nvs.push(nv),
            syn::Meta::List(list) if list.path.is_ident("window") => {
                let inner = Punctuated::<MetaNameValue, Token![,]>::parse_terminated
                    .parse2(list.tokens.clone())?;
                let (mut stride, mut pad, mut kernel, mut output) = (None, None, None, None);
                for nv in &inner {
                    let key = nv
                        .path
                        .get_ident()
                        .map(|i| i.to_string())
                        .unwrap_or_default();
                    // `stride`/`pad`/`kernel` take a const generic's name or a
                    // decimal literal; `output` names a runtime parameter, so
                    // it stays an identifier.
                    let named = match &nv.value {
                        Expr::Path(p) => p.path.get_ident().cloned(),
                        _ => None,
                    };
                    let literal = match &nv.value {
                        Expr::Lit(syn::ExprLit {
                            lit: syn::Lit::Int(i),
                            ..
                        }) => Some(i.base10_digits().to_string()),
                        _ => None,
                    };
                    let scalar = || match (&named, &literal) {
                        (Some(id), _) => Ok(id.to_string()),
                        (None, Some(lit)) => Ok(lit.clone()),
                        (None, None) => Err(syn::Error::new_spanned(
                            &nv.value,
                            "a `window(...)` stride, pad or kernel must be a const \
                             generic's name or a decimal integer literal",
                        )),
                    };
                    match key.as_str() {
                        "stride" => stride = Some(scalar()?),
                        "pad" => pad = Some(scalar()?),
                        "kernel" => kernel = Some(scalar()?),
                        "output" => {
                            output = Some(named.ok_or_else(|| {
                                syn::Error::new_spanned(
                                    &nv.value,
                                    "a `window(...)` `output` must name the output's runtime \
                                     extent parameter, not a literal",
                                )
                            })?)
                        }
                        other => {
                            return Err(syn::Error::new_spanned(
                                &nv.path,
                                format!(
                                    "unknown `window(...)` key `{other}` (expected `stride`, \
                                     `pad`, `kernel` or `output`)"
                                ),
                            ));
                        }
                    }
                }
                match (stride, kernel, output) {
                    // `pad` is optional: most pools have no padding const, and
                    // padding does not enter the receptive field anyway.
                    (Some(s), Some(k), Some(o)) => window = Some((s, pad, k, o)),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &list,
                            "`window(...)` needs `stride`, `kernel` and `output` (`pad` is \
                             optional): the receptive field is \
                             `(block - 1) * stride + kernel`, and `output` names the output \
                             axis whose block this one resolves against -- this axis's own \
                             extent never appears in the output",
                        ));
                    }
                }
            }
            syn::Meta::List(list) => {
                return Err(syn::Error::new_spanned(
                    &list,
                    "`#[tile(...)]` takes `key = value` pairs, bare flags and `window(...)`",
                ));
            }
        }
    }
    for nv in nvs {
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
        // `block` additionally takes a decimal literal, for an axis the kernel
        // steps one element at a time and so has no `BLOCK_*` const for. The
        // reduction family is the case: `row = program_id(Axis::X)` handles one
        // row per program, and there is no `BLOCK_OUTER` to name
        // (teenygrad-1tl.9). It is documentation either way -- `resolve_inputs`
        // takes an axis's block from the propagated output tile, never from
        // this name -- so a larger tile is still resolved correctly.
        if key == "block" {
            if let Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(i),
                ..
            }) = &nv.value
            {
                block = Some(i.base10_digits().to_string());
                continue;
            }
        }
        let Expr::Path(p) = &nv.value else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                "`#[tile(...)]` values must be bare identifiers (except `name`, a string literal, \
                 and `block`, which also takes a decimal integer literal)",
            ));
        };
        let Some(id) = p.path.get_ident().cloned() else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                "`#[tile(...)]` values must be a single identifier",
            ));
        };
        match key.as_str() {
            "block" => block = Some(id.to_string()),
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
        window,
        reduce,
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

/// A kernel's declared accumulation loop (teenygrad-1nr.18.3).
///
/// Metadata only: this drives the generated `tile_spec()`'s
/// [`TileLoopSpec`](teeny_core::model::TileLoopSpec) and nothing else. The
/// kernel keeps its own hand-written loop.
///
/// Generating the loop was considered and rejected for now -- see this
/// issue's design notes. The short version is that wrapping a kernel body in
/// a generated loop puts the body's trailing `T::store` *inside* the loop,
/// and delimiting "loop part" from "epilogue" needs markers in the body,
/// which is exactly what `84ca6eedf` was reverted for. A kernel whose input
/// tile varies per iteration (conv2d's `x`, read at offsets depending on
/// `(c_in, kh, kw)`) also cannot use the `In<Tile<..>>` form at all, since
/// the prelude loads such a parameter once, up front.
struct TileLoopArgs {
    /// Names of the `{NAME}: i32` params / `const {NAME}: i32` generics whose
    /// values together determine the trip count. A list of names rather than
    /// one param because a real trip count mixes them: conv2d's is
    /// `(C_IN / G) * KH * KW`.
    trip_count: Vec<Ident>,
    /// One entry per carried accumulator: the variable's name in the body, and
    /// the consts giving its shape in dimension order.
    carries: Vec<(Ident, Vec<String>)>,
    /// The loop's real, *evaluable* trip count, from `count = <expr>`.
    ///
    /// Separate from `trip_count` because that is a list of names and is
    /// documented as not being a formula -- conv2d's factors are
    /// `[C_IN, G, KH, KW]` but its count is `(C_IN / G) * KH * KW`, which no
    /// multiplication of the list produces. Generation needs the expression;
    /// the spec keeps the names (teenygrad-y8aa).
    count: Option<syn::Expr>,
    /// The loop's own axes, outermost to innermost, from
    /// `axes = [name = extent, ..]`.
    ///
    /// Present, these do two jobs: the product of the extents *is* the trip
    /// count (so `count` is not needed), and each name is bound inside the
    /// loop by the generated decode. conv2d writes that decode by hand as
    /// `kw = idx % KW; kh = idx / KW % KH; c_in_local = idx / (KW * KH)`,
    /// which is the same innermost-first arithmetic the flat `program_id`
    /// decode already generates -- Option D of teenygrad-1nr.18.3's analysis,
    /// placeable now that the loop itself is generated.
    axes: Vec<(Ident, syn::Expr)>,
    /// `true` when the bare `generate` flag is present: the macro emits the
    /// carry initialisation, the loop around the author's body, and the store
    /// after it.
    ///
    /// Opt-in so that the nine kernels already declaring a loop keep their
    /// metadata-only behaviour untouched (teenygrad-1nr.18.3 landed that, and
    /// it is what the rungs blocked on it needed).
    generate: bool,
}

/// Reads a carry's shape out of `key = [A, 1, B]`: each entry is a const name
/// or a decimal integer literal, matching
/// [`TileCarryBinding::shape_consts`](teeny_core::model::TileCarryBinding::shape_consts).
fn parse_shape_array(nv: &MetaNameValue) -> Result<Vec<String>, syn::Error> {
    let Expr::Array(array) = &nv.value else {
        return Err(syn::Error::new_spanned(
            &nv.value,
            "expected a bracketed shape, e.g. `[BLOCK_OW]` or `[1]`",
        ));
    };
    array
        .elems
        .iter()
        .map(|e| match e {
            Expr::Path(p) => p
                .path
                .get_ident()
                .map(Ident::to_string)
                .ok_or_else(|| syn::Error::new_spanned(e, "expected a single identifier")),
            Expr::Lit(syn::ExprLit {
                lit: syn::Lit::Int(i),
                ..
            }) => Ok(i.base10_digits().to_string()),
            other => Err(syn::Error::new_spanned(
                other,
                "a shape entry must be a const name or an integer literal",
            )),
        })
        .collect()
}

/// Reads the bracketed list out of `key = [A, B, C]`.
fn parse_ident_array(nv: &MetaNameValue) -> Result<Vec<Ident>, syn::Error> {
    let Expr::Array(array) = &nv.value else {
        return Err(syn::Error::new_spanned(
            &nv.value,
            "expected a bracketed list of names, e.g. `[BLOCK_OW]`",
        ));
    };
    array
        .elems
        .iter()
        .map(|e| match e {
            Expr::Path(p) => p
                .path
                .get_ident()
                .cloned()
                .ok_or_else(|| syn::Error::new_spanned(e, "expected a single identifier")),
            other => Err(syn::Error::new_spanned(
                other,
                "expected a single identifier",
            )),
        })
        .collect()
}

/// Parse a kernel's `#[tile_loop(trip_count = [..])]` and
/// `#[tile_carry(name = [..], ..)]` attributes. `None` when it declares no loop.
/// Parse a kernel's `#[tile_grid(order = [A, B, ..])]`, naming the grid axes
/// outermost to innermost as the body's `pid` decode actually produces them.
///
/// `grid_spec()` is otherwise built from the single `Out` parameter's axes in
/// *tensor dim* order, which is the decode order for every kernel whose output
/// dims and grid happen to agree -- all of them until `transpose_2d_forward`.
/// A transpose permutes the two: its output is `[N, M]` while its body decodes
/// `pid_m` outer (`pid / num_pid_n`), and `GridSpec::axes` is documented as
/// mattering, outermost to innermost. Declaration order cannot serve both,
/// because an axis's `dims` entry comes from its position
/// (teenygrad-1tl.5).
///
/// The list is *the grid axes*, so a subset is meaningful: an axis the body
/// covers with a loop rather than the grid is simply left out.
/// `batch_norm_normalize_forward` runs one program per channel and walks `N`
/// in a `while` loop, so its grid is `[C]` while its output is `[N, C]`
/// (teenygrad-1tl.8).
/// What a kernel says about its launch grid.
enum TileGridArgs {
    /// The grid axes, outermost to innermost.
    Order(Vec<Ident>),
    /// The `pid` decode is not a mixed-radix decode of the output's axes, so
    /// no `GridSpec` can describe it and none is emitted.
    Swizzled,
}

fn parse_tile_grid_order(attrs: &[syn::Attribute]) -> Result<Option<TileGridArgs>, syn::Error> {
    let Some(attr) = attrs.iter().find(|a| a.path().is_ident("tile_grid")) else {
        return Ok(None);
    };
    // `syn::Meta`, not `MetaNameValue`, so that the bare `swizzled` flag parses
    // alongside `order = [..]` -- the same reason `#[tile(.., reduce)]` does.
    let inner = attr.parse_args_with(
        syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
    )?;
    let mut out = None;
    for meta in &inner {
        match meta {
            syn::Meta::Path(path) if path.is_ident("swizzled") => {
                out = Some(TileGridArgs::Swizzled);
            }
            syn::Meta::NameValue(nv) if nv.path.is_ident("order") => {
                let Expr::Array(arr) = &nv.value else {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`order` takes a list of axis names, e.g. `order = [M, N]`",
                    ));
                };
                let mut names = Vec::new();
                for e in &arr.elems {
                    let Expr::Path(path) = e else {
                        return Err(syn::Error::new_spanned(
                            e,
                            "each `order` entry names one grid axis",
                        ));
                    };
                    names.push(path.path.get_ident().cloned().ok_or_else(|| {
                        syn::Error::new_spanned(e, "each `order` entry is a single identifier")
                    })?);
                }
                out = Some(TileGridArgs::Order(names));
            }
            other => {
                return Err(syn::Error::new_spanned(
                    other,
                    "unknown `#[tile_grid(...)]` argument (expected `order = [..]` or the bare \
                     flag `swizzled`)",
                ));
            }
        }
    }
    Ok(out)
}

/// Builds the three pieces of a wrapper-generated accumulation loop: the carry
/// initialisations, the loop itself around the author's body, and the single
/// store after it (teenygrad-y8aa).
///
/// Scoped deliberately to the unambiguous case -- one carry, one `In<Tile>`,
/// one `Out<Tile>` -- and it errors rather than guessing otherwise. Flash
/// attention has three carries and two outputs, so pairing carry to output
/// needs a declaration that does not exist yet; nothing here pretends to know
/// which carry belongs to which output.
///
/// Carries initialise to zeros. A non-zero initial value is real -- flash
/// attention's `m_i` starts at negative infinity -- and needs syntax of its
/// own, so such a kernel simply does not pass `generate` yet.
#[allow(clippy::type_complexity)]
fn generated_loop(
    l: &TileLoopArgs,
    hw_ident: &Ident,
    tile_in_params: &[(&Ident, syn::Type, &[TileAttrArgs])],
    tile_out_params: &[(&Ident, syn::Type, &[TileAttrArgs])],
    loop_scalars: &[(&Ident, syn::Type, Vec<(syn::Expr, syn::Expr)>)],
    loop_tiles: &[(&Ident, syn::Type, Vec<(syn::Expr, syn::Expr)>, Vec<usize>)],
    input: &ItemFn,
) -> Result<(Vec<syn::Stmt>, syn::Stmt, syn::Stmt), syn::Error> {
    let span = input.sig.ident.span();
    if l.carries.len() != 1 {
        return Err(syn::Error::new(
            span,
            format!(
                "`#[tile_loop(generate)]` supports exactly one carry for now, found {}: pairing \
                 several carries to several outputs needs a declaration that does not exist yet \
                 (teenygrad-y8aa)",
                l.carries.len()
            ),
        ));
    }
    if tile_out_params.len() != 1 {
        return Err(syn::Error::new(
            span,
            format!(
                "`#[tile_loop(generate)]` supports exactly one `Out<Tile<..>>` for now, found \
                 {}: the generated store needs one unambiguous destination (teenygrad-y8aa)",
                tile_out_params.len()
            ),
        ));
    }

    let (carry, shape) = &l.carries[0];
    let dims: Vec<TokenStream2> = shape
        .iter()
        .map(|d| {
            d.parse()
                .expect("a carry shape entry is an identifier or an integer literal")
        })
        .collect();
    let (out_ident, out_dtype, _) = &tile_out_params[0];
    // `axes` supplies the count as the product of its extents, so a kernel
    // declaring axes needs no separate `count`. Parenthesised per factor: an
    // extent is an arbitrary expression and conv2d's is `(C_IN / G)`.
    let count: TokenStream2 = if l.axes.is_empty() {
        let c = l
            .count
            .as_ref()
            .expect("checked in parse_tile_loop_attrs: generate requires axes or count");
        quote! { #c }
    } else {
        // Folded, not `#(..)*`-joined: quote's repetition takes no `*`
        // separator, so an unseparated join would emit `(C_IN / G) (KH) (KW)`
        // and parse as function calls.
        l.axes
            .iter()
            .map(|(_, e)| quote! { (#e) })
            .reduce(|acc, f| quote! { #acc * #f })
            .expect("axes is non-empty, checked when parsed")
    };

    // Option D: decode the flat loop index into one binding per declared axis,
    // innermost-first, exactly as the flat `program_id` decode above does for
    // the grid. conv2d writes this by hand today.
    let mut decode: Vec<syn::Stmt> = Vec::new();
    if !l.axes.is_empty() {
        let rem = format_ident!("__tile_loop_rem");
        decode.push(
            syn::parse2(quote! { let mut #rem = __tile_loop_idx; })
                .expect("generated loop remainder is valid Rust"),
        );
        for (pos, (name, extent)) in l.axes.iter().enumerate().rev() {
            if pos == 0 {
                // The outermost axis takes whatever is left, so no final
                // division is emitted -- same as the grid decode.
                decode.push(
                    syn::parse2(quote! { let #name = #rem; })
                        .expect("generated outermost loop index is valid Rust"),
                );
            } else {
                decode.push(
                    syn::parse2(quote! { let #name = #rem % (#extent); })
                        .expect("generated loop index is valid Rust"),
                );
                decode.push(
                    syn::parse2(quote! { #rem = #rem / (#extent); })
                        .expect("generated loop remainder update is valid Rust"),
                );
            }
        }
    }

    let init: syn::Stmt = syn::parse2(quote! {
        let mut #carry = #hw_ident::zeros::<#out_dtype>(&[#(#dims),*]);
    })
    .expect("generated carry initialisation is valid Rust");

    // The author's body, verbatim, as the loop body, preceded by the decode so
    // the axis names are in scope for it.
    // Per-iteration scalar operands, loaded inside the loop and broadcast to the
    // carry's shape. The broadcast target is the carry rather than a declared
    // shape because that is what makes the product well-typed: the scalar
    // multiplies a tile that is accumulated into the carry, so they agree by
    // construction (teenygrad-y8aa).
    let mut scalar_loads: Vec<syn::Stmt> = Vec::new();
    for (name, dtype, axes) in loop_scalars {
        let offset = row_major_offset(axes);
        let load_stmt: syn::Stmt = syn::parse2(quote! {
            let #name = {
                let __tile_scalar_off = #hw_ident::arange(0, 1) + (#offset);
                #hw_ident::broadcast_to(
                    #hw_ident::load(
                        #name.add_offsets(__tile_scalar_off),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    ),
                    &[#(#dims),*],
                )
            };
        })
        .expect("generated scalar operand load is valid Rust");
        let _ = dtype;
        scalar_loads.push(load_stmt);
    }

    // Per-iteration tile operands: a whole tile read at loop-dependent
    // coordinates, with a boundary mask. The vector counterpart of the scalar
    // loads above, and the second half of constraint C1 (teenygrad-y8aa).
    // The prelude binds the blocked axis's range as `__tile_range` and the
    // output tile's own mask as `in_bounds`; the generated read needs both. The
    // multi-blocked-axis prelude names its ranges `__tile_range_<slot>`
    // instead, so a windowed read is restricted to one blocked axis until a
    // kernel needs otherwise (teenygrad-y8aa).
    let range_ident = format_ident!("__tile_range");
    let mask_ident = format_ident!("in_bounds");
    if !loop_tiles.is_empty() {
        let blocked = tile_out_params[0]
            .2
            .iter()
            .filter(|a| a.block.is_some())
            .count();
        if blocked != 1 {
            return Err(syn::Error::new(
                span,
                format!(
                    "`#[tile_loop_tile]` needs exactly one blocked axis on the output, found \
                     {blocked}: the windowed coordinate's base is that axis's range, and the \
                     multi-axis prelude names its ranges per slot (teenygrad-y8aa)"
                ),
            ));
        }
    }
    for (name, dtype, axes, bounds) in loop_tiles {
        let offset = row_major_offset(axes);

        // Bounds checks on the windowed coordinates. A scalar coordinate has to
        // be splatted into a tensor first, and `range * 0 + scalar` is how
        // conv2d does it -- its own comment records that a scalar `if`/
        // `continue` there trips a compiler phi-node bug, so the splat is load
        // bearing rather than stylistic.
        let mut checks: Vec<TokenStream2> = Vec::new();
        for &i in bounds {
            let (coord, extent) = &axes[i];
            let splat = quote! { (#range_ident * 0 + (#coord)) };
            checks.push(quote! { #splat.ge(0) & #splat.lt(#extent) });
        }
        let mask = checks
            .into_iter()
            .fold(quote! { #mask_ident }, |acc, c| quote! { #acc & (#c) });

        let load_stmt: syn::Stmt = syn::parse2(quote! {
            let #name = {
                let __tile_read_mask = #mask;
                #hw_ident::load(
                    #name.add_offsets(#offset),
                    Some(__tile_read_mask),
                    Some(#hw_ident::zeros::<#dtype>(&[#(#dims),*])),
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
            };
        })
        .expect("generated tile operand load is valid Rust");
        scalar_loads.push(load_stmt);
    }

    let body_stmts: Vec<syn::Stmt> = input.block.stmts.clone();
    let loop_stmt: syn::Stmt = syn::parse2(quote! {
        for __tile_loop_idx in 0..(#count) {
            let _ = __tile_loop_idx;
            #(#decode)*
            #(#scalar_loads)*
            #(#body_stmts)*
        }
    })
    .expect("generated loop is valid Rust");

    let store_stmt: syn::Stmt = syn::parse2(quote! {
        #hw_ident::store(
            #out_ident.tensor,
            #carry,
            // Bare `Some`, not `::core::option::Option::Some`: the no_core
            // device source has no `core::option` path -- `Option` is the
            // prelude's own `#[lang = "Option"]` shim, used unqualified.
            // Generated *body* code is spliced into that source, unlike the
            // generated spec methods, which are host items and do use full
            // paths.
            Some(in_bounds),
            &[],
            None,
            None,
        );
    })
    .expect("generated store is valid Rust");

    Ok((vec![init], loop_stmt, store_stmt))
}

fn parse_tile_loop_attrs(attrs: &[syn::Attribute]) -> Result<Option<TileLoopArgs>, syn::Error> {
    let mut trip_count: Option<Vec<Ident>> = None;
    let mut carries: Vec<(Ident, Vec<String>)> = Vec::new();
    let mut count: Option<syn::Expr> = None;
    let mut loop_axes: Vec<(Ident, syn::Expr)> = Vec::new();
    let mut generate = false;

    for attr in attrs {
        let is_loop = attr.path().is_ident("tile_loop");
        let is_carry = attr.path().is_ident("tile_carry");
        if !is_loop && !is_carry {
            continue;
        }
        // `syn::Meta`, not `MetaNameValue`, so the bare `generate` flag parses
        // beside the `key = value` entries (teenygrad-y8aa).
        let metas = Punctuated::<syn::Meta, Token![,]>::parse_terminated
            .parse2(attr.meta.require_list()?.tokens.clone())?;
        let mut parsed: Vec<MetaNameValue> = Vec::new();
        for meta in metas {
            match meta {
                syn::Meta::Path(path) if is_loop && path.is_ident("generate") => generate = true,
                syn::Meta::NameValue(nv) => parsed.push(nv),
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        "expected `key = value`, or the bare flag `generate` on `#[tile_loop]`",
                    ));
                }
            }
        }
        for nv in &parsed {
            let key = nv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            if is_loop {
                if key == "axes" {
                    if !loop_axes.is_empty() {
                        return Err(syn::Error::new_spanned(
                            &nv.path,
                            "`axes` declared more than once",
                        ));
                    }
                    let syn::Expr::Array(arr) = &nv.value else {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "`axes` takes a list, e.g. `axes = [c_in = (C_IN / G), kh = KH]`",
                        ));
                    };
                    for e in &arr.elems {
                        let syn::Expr::Assign(a) = e else {
                            return Err(syn::Error::new_spanned(
                                e,
                                "each `axes` entry is `name = extent`, outermost to innermost",
                            ));
                        };
                        let syn::Expr::Path(np) = &*a.left else {
                            return Err(syn::Error::new_spanned(
                                &a.left,
                                "an axis's name is a single identifier",
                            ));
                        };
                        let name = np.path.get_ident().cloned().ok_or_else(|| {
                            syn::Error::new_spanned(
                                &a.left,
                                "an axis's name is a single identifier",
                            )
                        })?;
                        loop_axes.push((name, (*a.right).clone()));
                    }
                    if loop_axes.is_empty() {
                        return Err(syn::Error::new_spanned(
                            &nv.value,
                            "`axes` needs at least one entry",
                        ));
                    }
                    continue;
                }
                if key == "count" {
                    if count.is_some() {
                        return Err(syn::Error::new_spanned(
                            &nv.path,
                            "`count` declared more than once",
                        ));
                    }
                    count = Some(nv.value.clone());
                    continue;
                }
                if key != "trip_count" {
                    return Err(syn::Error::new_spanned(
                        &nv.path,
                        format!(
                            "unknown `#[tile_loop(...)]` key `{key}` (expected `trip_count`, \
                             `axes`, `count`, or the bare flag `generate`)"
                        ),
                    ));
                }
                if trip_count.is_some() {
                    return Err(syn::Error::new_spanned(
                        &nv.path,
                        "`trip_count` declared more than once",
                    ));
                }
                let names = parse_ident_array(nv)?;
                if names.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "`trip_count` needs at least one name; a loop with no factors has no \
                         trip count to describe",
                    ));
                }
                trip_count = Some(names);
            } else {
                let Some(name) = nv.path.get_ident().cloned() else {
                    return Err(syn::Error::new_spanned(
                        &nv.path,
                        "a carry's name must be a single identifier",
                    ));
                };
                let shape = parse_shape_array(nv)?;
                if shape.is_empty() {
                    return Err(syn::Error::new_spanned(
                        &nv.value,
                        "a carry needs at least one shape const: `KernelTileSpec::validate` \
                         rejects a carry that declares no shape",
                    ));
                }
                carries.push((name, shape));
            }
        }
    }

    match (trip_count, carries.is_empty()) {
        (None, true) => Ok(None),
        (None, false) => Err(syn::Error::new_spanned(
            attrs
                .iter()
                .find(|a| a.path().is_ident("tile_carry"))
                .expect("carries is non-empty, so a #[tile_carry] was seen"),
            "`#[tile_carry(...)]` needs a `#[tile_loop(trip_count = [..])]` alongside it: a \
             carry without a loop to carry it across means nothing",
        )),
        (Some(_), true) => Err(syn::Error::new_spanned(
            attrs
                .iter()
                .find(|a| a.path().is_ident("tile_loop"))
                .expect("trip_count is Some, so a #[tile_loop] was seen"),
            "`#[tile_loop(...)]` needs at least one `#[tile_carry(name = [..])]`: a loop that \
             carries nothing is not an accumulation loop, and teenygrad-1nr.18.3 only describes \
             accumulating loops -- an independent walk belongs in the grid instead",
        )),
        (Some(trip_count), false) => {
            // `generate` needs an evaluable count: `trip_count`'s names cannot
            // produce one, by its own contract (teenygrad-y8aa).
            if generate && count.is_none() && loop_axes.is_empty() {
                return Err(syn::Error::new_spanned(
                    attrs
                        .iter()
                        .find(|a| a.path().is_ident("tile_loop"))
                        .expect("trip_count is Some, so a #[tile_loop] was seen"),
                    "`#[tile_loop(generate)]` needs either `axes = [name = extent, ..]` or `count = <expr>`: `trip_count` is a list of names and is documented as not being a formula, so it cannot be evaluated -- conv2d's factors are [C_IN, G, KH, KW] but its count is (C_IN / G) * KH * KW. `axes` is preferred: its extents multiply to the count and their names are bound by the generated decode.",
                ));
            }
            Ok(Some(TileLoopArgs {
                trip_count,
                carries,
                count,
                axes: loop_axes,
                generate,
            }))
        }
    }
}

/// Strip a `#[tile(...)]` attribute from a parameter's attribute list, if
/// present -- it is host-only metadata (like `Tile` itself), never a real
/// attribute macro registered anywhere, so it must not reach the
/// regenerated device/host signatures this macro emits.
fn strip_tile_attr(pt: &mut PatType) {
    pt.attrs.retain(|a| {
        !a.path().is_ident("tile")
            && !a.path().is_ident("tile_loop_scalar")
            && !a.path().is_ident("tile_loop_tile")
    });
}

/// Row-major flat offset from `(index, extent)` pairs, outermost first.
///
/// `acc * extent + index`, folded from the outermost index. The outermost
/// extent never enters the result -- it is declared to document the layout --
/// which is the same reason the prelude's `stride_of` is `None` for the
/// innermost axis. conv2d writes this by hand as
/// `(b * C_IN + c_in) * H * W + ih * W` plus the innermost coordinate
/// (teenygrad-y8aa).
fn row_major_offset(axes: &[(syn::Expr, syn::Expr)]) -> TokenStream2 {
    let mut offset = {
        let (first_idx, _) = &axes[0];
        quote! { #first_idx }
    };
    for (idx, extent) in axes.iter().skip(1) {
        // Index first: `i + acc * e`, not `acc * e + i`. Same value, but the
        // innermost coordinate of a windowed read is a *tensor* while the outer
        // fold is scalar, and only `Tensor + i32` has an impl -- `i32 + Tensor`
        // does not. conv2d writes it the same way round for the same reason:
        // `iw_range + ((b * C_IN + c_in) * H * W + ih * W)`.
        offset = quote! { (#idx) + (#offset) * (#extent) };
    }
    offset
}

/// Reads a parameter's `#[tile_loop_tile(index = [..], bounds = [..])]`.
///
/// The vector sibling of [`parse_tile_loop_scalar`]: a whole tile read per
/// iteration at loop-dependent coordinates, with a boundary mask, rather than
/// one broadcast element. conv2d's `x`:
///
/// ```text
/// let ih       = oh * STRIDE_H + kh - PAD_H;
/// let iw_range = ow_range * STRIDE_W + kw - PAD_W;
/// let mask     = ow_mask & ih_t.ge(0) & ih_t.lt(H) & iw_range.ge(0) & iw_range.lt(W);
/// T::load(x_ptr.add_offsets(iw_range + ((b * C_IN + c_in) * H * W + ih * W)), Some(mask), ..)
/// ```
///
/// Coordinates are full expressions rather than names, deliberately. A windowed
/// coordinate is `base * stride + tap - pad`, mixing a grid index, a loop index
/// and two consts, and every one of those is already in scope where the
/// generated read sits -- so taking the expression keeps the macro out of the
/// business of inferring which loop axis pairs with which window, which would
/// be ambiguous the moment two axes shared a kernel const.
///
/// `bounds` lists the positions whose coordinate needs a `ge(0) & lt(extent)`
/// check -- the windowed ones. A plain axis indexed by a grid index is in bounds
/// by construction and is left out, exactly as conv2d masks only H and W.
fn parse_tile_loop_tile(
    pt: &PatType,
) -> Result<Option<(Vec<(syn::Expr, syn::Expr)>, Vec<usize>)>, syn::Error> {
    let Some(attr) = pt
        .attrs
        .iter()
        .find(|a| a.path().is_ident("tile_loop_tile"))
    else {
        return Ok(None);
    };
    let inner = attr.parse_args_with(
        syn::punctuated::Punctuated::<MetaNameValue, Token![,]>::parse_terminated,
    )?;
    let mut axes: Vec<(syn::Expr, syn::Expr)> = Vec::new();
    let mut bounds: Vec<usize> = Vec::new();
    for nv in &inner {
        let key = nv
            .path
            .get_ident()
            .map(|i| i.to_string())
            .unwrap_or_default();
        let Expr::Array(arr) = &nv.value else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                "`#[tile_loop_tile(...)]` values are lists",
            ));
        };
        match key.as_str() {
            "index" => {
                for e in &arr.elems {
                    let Expr::Assign(a) = e else {
                        return Err(syn::Error::new_spanned(
                            e,
                            "each `index` entry is `coord_expr = extent_expr`, outermost first",
                        ));
                    };
                    axes.push(((*a.left).clone(), (*a.right).clone()));
                }
            }
            "bounds" => {
                for e in &arr.elems {
                    let Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Int(i),
                        ..
                    }) = e
                    else {
                        return Err(syn::Error::new_spanned(
                            e,
                            "each `bounds` entry is the integer position of an axis in `index`",
                        ));
                    };
                    bounds.push(i.base10_parse()?);
                }
            }
            other => {
                return Err(syn::Error::new_spanned(
                    &nv.path,
                    format!(
                        "unknown `#[tile_loop_tile(...)]` key `{other}` (expected `index` or \
                         `bounds`)"
                    ),
                ));
            }
        }
    }
    if axes.is_empty() {
        return Err(syn::Error::new_spanned(
            attr,
            "`#[tile_loop_tile(index = [..])]` needs at least one axis",
        ));
    }
    if let Some(&bad) = bounds.iter().find(|&&b| b >= axes.len()) {
        return Err(syn::Error::new_spanned(
            attr,
            format!(
                "`bounds` names position {bad}, but `index` declares {} axes",
                axes.len()
            ),
        ));
    }
    Ok(Some((axes, bounds)))
}

/// Reads a parameter's `#[tile_loop_scalar(index = [name = extent, ..])]`.
///
/// Declares an operand read as ONE element per loop iteration at a
/// loop-dependent flat index, then broadcast -- conv2d's weight, whose whole
/// per-iteration handling is
///
/// ```text
/// let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
/// let w_off = T::arange(0, 1) + w_idx;
/// T::broadcast_to(T::load(w_ptr.add_offsets(w_off), ..), &[BLOCK_OW])
/// ```
///
/// The entries are `(index, extent)` pairs, outermost first, describing the
/// operand's layout: `[c_out = C_OUT, c_in_local = (C_IN / G), kh = KH,
/// kw = KW]` for a weight laid out `[C_OUT, C_IN/G, KH, KW]`. The offset is
/// then the row-major dot product, which is the same stride derivation the
/// prelude already does for a tile parameter.
///
/// An index may name anything in scope inside the loop, which is why this works
/// at all: `c_out` comes from the grid decode and `c_in_local`/`kh`/`kw` from
/// the loop decode, and by the time the generated load runs both are bound
/// (teenygrad-y8aa).
fn parse_tile_loop_scalar(pt: &PatType) -> Result<Option<Vec<(syn::Expr, syn::Expr)>>, syn::Error> {
    let Some(attr) = pt
        .attrs
        .iter()
        .find(|a| a.path().is_ident("tile_loop_scalar"))
    else {
        return Ok(None);
    };
    let inner = attr.parse_args_with(
        syn::punctuated::Punctuated::<MetaNameValue, Token![,]>::parse_terminated,
    )?;
    let mut axes: Vec<(syn::Expr, syn::Expr)> = Vec::new();
    for nv in &inner {
        if !nv.path.is_ident("index") {
            return Err(syn::Error::new_spanned(
                &nv.path,
                "unknown `#[tile_loop_scalar(...)]` key (expected `index`)",
            ));
        }
        let Expr::Array(arr) = &nv.value else {
            return Err(syn::Error::new_spanned(
                &nv.value,
                "`index` takes a list, e.g. `index = [c_out = C_OUT, kh = KH, kw = KW]`",
            ));
        };
        for e in &arr.elems {
            let Expr::Assign(a) = e else {
                return Err(syn::Error::new_spanned(
                    e,
                    "each `index` entry is `index_expr = extent_expr`, outermost first",
                ));
            };
            axes.push(((*a.left).clone(), (*a.right).clone()));
        }
    }
    if axes.is_empty() {
        return Err(syn::Error::new_spanned(
            attr,
            "`#[tile_loop_scalar(index = [..])]` needs at least one entry",
        ));
    }
    Ok(Some(axes))
}

// ── Macro implementation ──────────────────────────────────────────────────────

pub fn tiled_kernel(attrs: TokenStream, item: TokenStream) -> TokenStream {
    let kernel_attrs = match parse_kernel_attrs(attrs) {
        Ok(a) => a,
        Err(e) => return e.to_compile_error().into(),
    };
    let input = parse_macro_input!(item as ItemFn);
    let tile_loop = match parse_tile_loop_attrs(&input.attrs) {
        Ok(l) => l,
        Err(e) => return e.to_compile_error().into(),
    };
    let grid_order = match parse_tile_grid_order(&input.attrs) {
        Ok(o) => o,
        Err(e) => return e.to_compile_error().into(),
    };
    // `loop_spec` for the generated `tile_spec()`, or `None` when the kernel
    // declares no accumulation loop (teenygrad-1nr.18.3).
    let loop_spec_tokens: TokenStream2 = match &tile_loop {
        None => quote! { ::core::option::Option::None },
        Some(l) => {
            let carries = l.carries.iter().map(|(name, shape)| {
                let name_str = name.to_string();
                let shape_strs: Vec<String> = shape.clone();
                quote! {
                    ::teeny_core::model::TileCarryBinding {
                        name: #name_str,
                        shape_consts: &[ #(#shape_strs),* ],
                    }
                }
            });
            let factors: Vec<String> = l.trip_count.iter().map(Ident::to_string).collect();
            quote! {
                ::core::option::Option::Some(::teeny_core::model::TileLoopSpec {
                    carries: &[ #(#carries),* ],
                    trip_count_factors: &[ #(#factors),* ],
                })
            }
        }
    };
    let fn_ident = input.sig.ident.clone();
    let fn_name_str = fn_ident.to_string();
    let vis = &input.vis;
    // `#[tile_loop]`/`#[tile_carry]` are host-only metadata, like `#[tile]` on a
    // parameter: no attribute macro is registered for them anywhere, so they
    // must not reach the signatures this macro re-emits.
    let attrs: Vec<&syn::Attribute> = input
        .attrs
        .iter()
        .filter(|a| {
            !a.path().is_ident("tile_loop")
                && !a.path().is_ident("tile_carry")
                && !a.path().is_ident("tile_grid")
        })
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

    // teenygrad-1nr.18.1: a `Tile` parameter carries one `#[tile(...)]` per
    // real axis, outermost first. One axis is the single-flat-axis case the
    // auto-prelude has always handled; N is the generalization.
    // teenygrad-y8aa: a `#[tile_loop_scalar]` operand is read once per
    // iteration at a loop-dependent index, so it must NOT get the prelude's
    // load-once-up-front treatment. It is collected separately and loaded by
    // the generated loop instead -- which is constraint C1 for the scalar case.
    let loop_scalars: Vec<(&Ident, Type, Vec<(syn::Expr, syn::Expr)>)> = {
        let mut out = Vec::new();
        for pt in fn_inputs.iter() {
            let axes = match parse_tile_loop_scalar(pt) {
                Ok(Some(a)) => a,
                Ok(None) => continue,
                Err(e) => return e.to_compile_error().into(),
            };
            let Some(dtype) = in_tile_dtype(&pt.ty, &hw_ident) else {
                return syn::Error::new_spanned(
                    &pt.ty,
                    "`#[tile_loop_scalar]` applies to an `In<Tile<..>>` parameter: it replaces \
                     the prelude's load with a per-iteration one, and only a tile parameter has \
                     a prelude load to replace (teenygrad-y8aa)",
                )
                .to_compile_error()
                .into();
            };
            let Pat::Ident(pi) = &*pt.pat else { continue };
            out.push((&pi.ident, dtype, axes));
        }
        out
    };
    let loop_tiles: Vec<(&Ident, Type, Vec<(syn::Expr, syn::Expr)>, Vec<usize>)> = {
        let mut out = Vec::new();
        for pt in fn_inputs.iter() {
            let (axes, bounds) = match parse_tile_loop_tile(pt) {
                Ok(Some(a)) => a,
                Ok(None) => continue,
                Err(e) => return e.to_compile_error().into(),
            };
            let Some(dtype) = in_tile_dtype(&pt.ty, &hw_ident) else {
                return syn::Error::new_spanned(
                    &pt.ty,
                    "`#[tile_loop_tile]` applies to an `In<Tile<..>>` parameter: it replaces the \
                     prelude's load with a per-iteration one (teenygrad-y8aa)",
                )
                .to_compile_error()
                .into();
            };
            let Pat::Ident(pi) = &*pt.pat else { continue };
            out.push((&pi.ident, dtype, axes, bounds));
        }
        out
    };
    let is_loop_scalar = |id: &Ident| {
        loop_scalars.iter().any(|(n, _, _)| *n == id)
            || loop_tiles.iter().any(|(n, _, _, _)| *n == id)
    };

    let tile_in_params: Vec<(&Ident, Type, &[TileAttrArgs])> = fn_inputs
        .iter()
        .zip(tile_attrs.iter())
        .filter_map(|(pt, attrs)| {
            let dtype = in_tile_dtype(&pt.ty, &hw_ident)?;
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            if is_loop_scalar(&pi.ident) {
                return None;
            }
            Some((&pi.ident, dtype, attrs.as_slice()))
        })
        .collect();
    let tile_out_params: Vec<(&Ident, Type, &[TileAttrArgs])> = fn_inputs
        .iter()
        .zip(tile_attrs.iter())
        .filter_map(|(pt, attrs)| {
            let dtype = out_tile_dtype(&pt.ty, &hw_ident)?;
            let Pat::Ident(pi) = &*pt.pat else {
                return None;
            };
            Some((&pi.ident, dtype, attrs.as_slice()))
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
            // A decimal literal names no const generic and is exempt: it
            // says the kernel steps this axis one element at a time
            // (teenygrad-1tl.9). The span is the axis's extent, a real
            // identifier, since the block is now a plain `String`.
            if let Some(block) = &axis.block
                && block.parse::<u32>().is_err()
                && !const_params.iter().any(|cp| cp.ident == *block)
            {
                return syn::Error::new_spanned(
                    &axis.extent,
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
    let all_tile_param_attrs: Vec<&[TileAttrArgs]> = tile_in_params
        .iter()
        .chain(tile_out_params.iter())
        .map(|(_, _, a)| *a)
        .collect();
    let has_explicit_tile_attr =
        !all_tile_param_attrs.is_empty() && all_tile_param_attrs.iter().all(|a| !a.is_empty());
    if !all_tile_param_attrs.is_empty()
        && all_tile_param_attrs.iter().any(|a| !a.is_empty())
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
    // teenygrad-1nr.18.1: every `Tile` parameter must describe the *same*
    // axes, in the same order. This generalizes the previous rule (all
    // parameters share one `#[tile(block=..,extent=..)]`) from one axis to
    // N, and keeps the prelude's job simple: one grid decode serves every
    // parameter, because they all sit on the same axes.
    //
    // Exactly one of those axes may carry `block = ..`. A second blocked
    // axis would make the index tile genuinely 2-D (`expand_dims` plus
    // `Tensor<i32, 2>` bounds instead of the `Tensor<i32, 1>` every kernel
    // declares today), which is a separate ABI change -- see this issue's
    // own scope note.
    if has_explicit_tile_attr {
        // The *output* defines the kernel's axes: the grid is sized to cover
        // it, and every other parameter's axes are resolved against it by
        // name. An input may declare a subset -- a `(C,)` bias against an
        // `[N, C]` activation -- which is broadcasting, and the prelude
        // handles it by loading that operand's single element and
        // broadcasting it across the blocked axis.
        let Some((_, _, first)) = tile_out_params.first() else {
            return syn::Error::new_spanned(
                &input.sig,
                "a kernel with `#[tile(...)]`-tagged `Tile` parameters needs an \
                 `Out<Tile<..>>` parameter: the output's axes are what the grid covers \
                 and what every input's axes resolve against",
            )
            .to_compile_error()
            .into();
        };
        let first: &[TileAttrArgs] = first;
        for (ident, _, other) in tile_in_params.iter().chain(tile_out_params.iter()) {
            for axis in other.iter() {
                let known = first
                    .iter()
                    .any(|a| a.extent == axis.extent && a.block == axis.block && a.dim == axis.dim);
                if !known {
                    return syn::Error::new_spanned(
                        &axis.extent,
                        format!(
                            "`{ident}` declares an axis the output does not, so the prelude \
                             cannot place it. An input axis must either match an output axis \
                             by name, or be left off entirely (which broadcasts it). An axis \
                             related to an output axis by a stride/padding window is \
                             teenygrad-1nr.18.2, not this prelude"
                        ),
                    )
                    .to_compile_error()
                    .into();
                }
            }
        }
        for (ident, _, axes) in tile_out_params.iter() {
            if !axes.iter().any(|a| a.block.is_some()) {
                return syn::Error::new_spanned(
                    &input.sig,
                    format!(
                        "`{ident}` is an output, so it must declare the block-tiled axis -- a \
                         broadcast output would have several CTAs writing the same element"
                    ),
                )
                .to_compile_error()
                .into();
            }
        }
        let blocked: Vec<&TileAttrArgs> = first.iter().filter(|a| a.block.is_some()).collect();
        if blocked.is_empty() {
            return syn::Error::new_spanned(
                &first[0].extent,
                "an `In<Tile<..>>`/`Out<Tile<..>>` parameter needs one `#[tile(...)]` axis \
                 carrying `block = ..` -- a fully untiled `Tile` parameter has no tile to load",
            )
            .to_compile_error()
            .into();
        }
        // Several blocked axes are allowed since teenygrad-1nr.18.5: each
        // contributes its own range, broadcast into its own dimension. Such a
        // kernel must declare `T::BoolTensor: BitAnd<Output = T::BoolTensor>`,
        // because the per-axis bounds are conjoined into one mask.
    }

    let final_block = if tile_in_params.is_empty() && tile_out_params.is_empty() {
        input.block.as_ref().clone()
    } else {
        // Axes this kernel's tile parameters sit on, outermost first. The
        // explicit case takes them from `#[tile(...)]`; the fallback
        // synthesizes the one flat axis the hardcoded
        // `BLOCK_SIZE`/`n_elements` convention has always meant.
        let axes: Vec<TileAttrArgs> = if has_explicit_tile_attr {
            // The output's axes: the grid covers them, and every input's
            // axes were checked against them above.
            tile_out_params
                .first()
                .map(|(_, _, a)| a.to_vec())
                .expect("checked above: an explicit tile spec needs an Out<Tile<..>> param")
        } else {
            let Some(block_size) = const_params.iter().find(|cp| cp.ident == "BLOCK_SIZE") else {
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
            vec![TileAttrArgs {
                block: Some(block_size.ident.to_string()),
                extent: format_ident!("n_elements"),
                name: None,
                dim: None,
                window: None,
                // The implicit flat convention reduces nothing: it maps one
                // element to one element.
                reduce: false,
            }]
        };

        // Every name an axis refers to must really exist on this kernel.
        for axis in &axes {
            // A decimal literal names no const generic and is exempt: it
            // says the kernel steps this axis one element at a time
            // (teenygrad-1tl.9). The span is the axis's extent, a real
            // identifier, since the block is now a plain `String`.
            if let Some(block) = &axis.block
                && block.parse::<u32>().is_err()
                && !const_params.iter().any(|cp| cp.ident == *block)
            {
                return syn::Error::new_spanned(
                    &axis.extent,
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

        // Every axis carrying `block = ..`, in declaration order
        // (teenygrad-1nr.18.5). One is the common case; several make the tile
        // genuinely K-D.
        let blocked_positions: Vec<usize> = axes
            .iter()
            .enumerate()
            .filter(|(_, a)| a.block.is_some())
            .map(|(i, _)| i)
            .collect();
        let blocked_at = *blocked_positions
            .first()
            .expect("checked above: at least one axis carries `block = ..`");
        // Re-parsed into tokens: a block is a const generic's name or a decimal
        // literal, and a `String` in `quote!` would emit a string literal
        // (teenygrad-1tl.9).
        let block_ident: TokenStream2 = axes[blocked_at]
            .block
            .as_ref()
            .expect("filter found it")
            .parse()
            .expect("a block is an identifier or an integer literal");

        // How many CTAs cover each axis: a blocked axis is covered in
        // `cdiv(extent, block)` steps, an untiled one is one CTA per index.
        let axis_count = |a: &TileAttrArgs| -> TokenStream2 {
            let extent = &a.extent;
            match &a.block {
                Some(b) => {
                    let b: TokenStream2 = b
                        .parse()
                        .expect("a block is an identifier or an integer literal");
                    quote! { #hw_ident::cdiv(#extent, #b) }
                }
                None => quote! { #extent },
            }
        };

        // Each axis's CTA index is bound under a name the kernel body can
        // use: `#[tile(name = "C", ..)]` binds `tile_c`. A multi-axis kernel
        // almost always needs them -- `channel_bias_add_forward` indexes its
        // bias by the channel index, `conv2d_forward` needs `b`/`c_out`/`oh`
        // -- and the alternative is the body reaching for a generated name it
        // was never promised. For the blocked axis this is the *tile* index,
        // not an element offset; the elements are already in the loaded tile.
        let axis_idx_ident = |axis: &TileAttrArgs| {
            let label = axis
                .name
                .as_ref()
                .map(syn::LitStr::value)
                .unwrap_or_else(|| axis.extent.to_string());
            format_ident!("tile_{}", label.to_lowercase())
        };
        let idx_ident = |i: usize| axis_idx_ident(&axes[i]);

        // The single-axis case is emitted the way it always was. The general
        // decode below would be correct for it too -- it collapses to the
        // same arithmetic -- but every kernel's asm snapshot embeds `.loc`
        // line/column debug info, so routing them through it would rewrite
        // a pile of snapshots for a purely cosmetic change. Keeping the old
        // form is churn avoidance, not a correctness requirement; fold the
        // two paths together whenever re-recording those snapshots is
        // worth it.
        let mut stmts: Vec<syn::Stmt> = if axes.len() == 1 {
            let dim_ident = axes[0].dim.clone().unwrap_or_else(|| format_ident!("X"));
            let extent_ident = &axes[0].extent;
            syn::parse2::<syn::Block>(quote! {{
                let pid = #hw_ident::program_id(Axis::#dim_ident);
                let block_start = pid * #block_ident;
                let offsets = #hw_ident::arange(0, #block_ident) + block_start;
                let in_bounds = offsets.lt(#extent_ident);
            }})
            .expect("generated tile prelude is valid Rust")
            .stmts
        } else {
            let mut stmts: Vec<syn::Stmt> = Vec::new();

            // One flat `program_id` per hardware dim, decoded innermost-first
            // into a per-axis index -- the same decode `conv2d_forward` writes
            // by hand (`ow_tile = pid % num_ow_tiles; bco = pid / ...`).
            let mut dims_seen: Vec<String> = Vec::new();
            for axis in &axes {
                let d = axis
                    .dim
                    .as_ref()
                    .map(|d| d.to_string())
                    .unwrap_or_else(|| "X".to_string());
                if !dims_seen.contains(&d) {
                    dims_seen.push(d);
                }
            }
            for dim_name in &dims_seen {
                let dim_ident = format_ident!("{}", dim_name);
                let on_dim: Vec<usize> = axes
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| {
                        a.dim
                            .as_ref()
                            .map(|d| d.to_string())
                            .unwrap_or_else(|| "X".to_string())
                            == *dim_name
                    })
                    .map(|(i, _)| i)
                    .collect();
                let rem = format_ident!("__tile_rem_{}", dim_name.to_lowercase());
                stmts.push(
                    syn::parse2(quote! {
                        let mut #rem = #hw_ident::program_id(Axis::#dim_ident);
                    })
                    .expect("generated program_id statement is valid Rust"),
                );
                // Innermost varies fastest; the outermost takes what is left.
                for (pos, &i) in on_dim.iter().enumerate().rev() {
                    let idx = idx_ident(i);
                    if pos == 0 {
                        stmts.push(
                            syn::parse2(quote! { let #idx = #rem; })
                                .expect("generated outermost index is valid Rust"),
                        );
                    } else {
                        let count = axis_count(&axes[i]);
                        stmts.push(
                            syn::parse2(quote! {
                                let #idx = #rem % (#count);
                            })
                            .expect("generated index statement is valid Rust"),
                        );
                        stmts.push(
                            syn::parse2(quote! {
                                #rem = #rem / (#count);
                            })
                            .expect("generated remainder statement is valid Rust"),
                        );
                    }
                }
            }

            // Row-major contiguous strides, from the declared extents: the
            // stride of an axis is the product of every extent inside it. Same
            // arithmetic `conv2d_forward` writes by hand as `... * H * W + ... * W`.
            // `None` for the innermost axis, whose stride is 1 -- so the
            // generated source reads `range * C + c` rather than
            // `range * (1 * C) + c * (1)`, matching what the kernel author
            // would have written by hand.
            let stride_of = |i: usize| -> Option<TokenStream2> {
                let inner: Vec<&Ident> = axes[i + 1..].iter().map(|a| &a.extent).collect();
                inner
                    .split_first()
                    .map(|(head, rest)| quote! { #head #( * #rest)* })
            };

            // One range per blocked axis (teenygrad-1nr.18.5). With a single
            // blocked axis this is the familiar `arange(0, B) + idx * B`. With
            // several, each range is broadcast into its own dimension so the
            // tile is genuinely K-D: the outer gets `[B0, 1]`, the inner
            // `[1, B1]`, and the arithmetic below broadcasts them together.
            //
            // The ranges are expanded *before* comparing, not after, so the
            // masks broadcast without needing `expand_dims` over a
            // `BoolTensor` -- `expand_dims` is declared over `Tensor<D>`.
            let blocked_ranges: Vec<(usize, Ident)> = blocked_positions
                .iter()
                .enumerate()
                .map(|(slot, &i)| {
                    let blk: TokenStream2 = axes[i]
                        .block
                        .as_ref()
                        .expect("blocked_positions only holds blocked axes")
                        .parse()
                        .expect("a block is an identifier or an integer literal");
                    let idx = idx_ident(i);
                    let mut expr = quote! { #hw_ident::arange(0, #blk) + #idx * #blk };
                    if blocked_positions.len() > 1 {
                        // Ascending order: [B] -> expand at 1 -> [B, 1], and so
                        // on, leaving this axis's own dimension alone.
                        for d in 0..blocked_positions.len() {
                            if d != slot {
                                let d = d as i32;
                                expr = quote! { #hw_ident::expand_dims_i32(#expr, #d) };
                            }
                        }
                    }
                    // One blocked axis keeps the original binding name, so
                    // adding this feature rewrites no existing snapshot -- the
                    // same churn avoidance the single-axis prelude is written
                    // for. The suffixed form appears only where there really
                    // are several blocked axes.
                    let rng = if blocked_positions.len() == 1 {
                        format_ident!("__tile_range")
                    } else {
                        format_ident!("__tile_range_{}", slot)
                    };
                    stmts.push(
                        syn::parse2(quote! { let #rng = #expr; })
                            .expect("generated arange statement is valid Rust"),
                    );
                    (i, rng)
                })
                .collect();
            // Every blocked axis contributes a bound; with more than one they
            // are conjoined, which is why such a kernel must declare
            // `T::BoolTensor: BitAnd<Output = T::BoolTensor>`.
            let mask_expr = blocked_ranges
                .iter()
                .map(|(i, rng)| {
                    let extent = &axes[*i].extent;
                    quote! { #rng.lt(#extent) }
                })
                .reduce(|a, b| quote! { #a & #b })
                .expect("at least one axis carries `block = ..`");
            stmts.push(
                syn::parse2(quote! { let in_bounds = #mask_expr; })
                    .expect("generated mask statement is valid Rust"),
            );

            // The untiled axes contribute a scalar base; the blocked one
            // contributes the tensor. Adding the scalar to the tensor is how
            // `conv2d_forward` folds its own `b`/`c_out`/`oh` base in.
            let scalar_terms: Vec<TokenStream2> = axes
                .iter()
                .enumerate()
                .filter(|(i, _)| !blocked_positions.contains(i))
                .map(|(i, _)| {
                    let idx = idx_ident(i);
                    match stride_of(i) {
                        Some(stride) => quote! { #idx * (#stride) },
                        None => quote! { #idx },
                    }
                })
                .collect();
            let blocked_terms: Vec<TokenStream2> = blocked_ranges
                .iter()
                .map(|(i, rng)| match stride_of(*i) {
                    Some(stride) => quote! { #rng * (#stride) },
                    None => quote! { #rng },
                })
                .collect();
            let offsets_expr: TokenStream2 = if scalar_terms.is_empty() {
                quote! { #(#blocked_terms)+* }
            } else {
                quote! { #(#blocked_terms)+* + #(#scalar_terms)+* }
            };
            stmts.push(
                syn::parse2(quote! { let offsets = #offsets_expr; })
                    .expect("generated offsets statement is valid Rust"),
            );
            stmts
        };

        // Offsets are per parameter, because a parameter need not sit on
        // every axis. One that declares the blocked axis addresses a real
        // tile; one that leaves it off is a broadcast operand -- a `(C,)`
        // bias against an `[N, C]` activation -- and addresses the single
        // element this CTA needs, widened across the block after loading so
        // every `Tile` in the body has the same shape.
        let param_offsets = |param_axes: &[TileAttrArgs]| -> (TokenStream2, bool) {
            // No attributes at all means the implicit
            // `BLOCK_SIZE`/`n_elements` convention, which puts the parameter
            // on the kernel's one axis. Without this it would look like a
            // parameter that declares *no* axes, i.e. a broadcast operand.
            let param_axes = if param_axes.is_empty() {
                &axes[..]
            } else {
                param_axes
            };
            // A parameter on the kernel's full axis set reuses the shared
            // `offsets` binding rather than restating it -- which is every
            // parameter in the single-axis case, and the activations in a
            // broadcast one.
            let same_as_kernel = param_axes.len() == axes.len()
                && param_axes
                    .iter()
                    .zip(axes.iter())
                    .all(|(a, b)| a.block == b.block && a.extent == b.extent && a.dim == b.dim);
            if same_as_kernel {
                return (quote! { offsets }, false);
            }
            let stride_within = |i: usize| -> Option<TokenStream2> {
                let inner: Vec<&Ident> = param_axes[i + 1..].iter().map(|a| &a.extent).collect();
                inner
                    .split_first()
                    .map(|(head, rest)| quote! { #head #( * #rest)* })
            };
            // A parameter may sit on several blocked axes, each contributing
            // its own range (teenygrad-1nr.18.5). The ranges are named by the
            // *kernel's* blocked order, so a parameter declaring a subset still
            // picks up the right ones.
            let mut blocked_terms: Vec<TokenStream2> = Vec::new();
            let mut scalar_terms: Vec<TokenStream2> = Vec::new();
            for (i, axis) in param_axes.iter().enumerate() {
                let stride = stride_within(i);
                if axis.block.is_some() {
                    let slot = blocked_positions
                        .iter()
                        .position(|&k| axes[k].block == axis.block)
                        .unwrap_or(0);
                    // One blocked axis keeps the original binding name, so
                    // adding this feature rewrites no existing snapshot -- the
                    // same churn avoidance the single-axis prelude is written
                    // for. The suffixed form appears only where there really
                    // are several blocked axes.
                    let rng = if blocked_positions.len() == 1 {
                        format_ident!("__tile_range")
                    } else {
                        format_ident!("__tile_range_{}", slot)
                    };
                    blocked_terms.push(match stride {
                        Some(stride) => quote! { #rng * (#stride) },
                        None => quote! { #rng },
                    });
                } else {
                    let idx = axis_idx_ident(axis);
                    scalar_terms.push(match stride {
                        Some(stride) => quote! { #idx * (#stride) },
                        None => quote! { #idx },
                    });
                }
            }
            let blocked_term: Option<TokenStream2> = if blocked_terms.is_empty() {
                None
            } else {
                Some(quote! { #(#blocked_terms)+* })
            };
            match blocked_term {
                Some(blocked) if scalar_terms.is_empty() => (blocked, false),
                Some(blocked) => (quote! { #blocked + #(#scalar_terms)+* }, false),
                // No blocked axis: one element, so `arange(0, 1)` makes it a
                // tensor the pointer arithmetic can take.
                None if scalar_terms.is_empty() => (quote! { #hw_ident::arange(0, 1) }, true),
                None => (
                    quote! { #hw_ident::arange(0, 1) + #(#scalar_terms)+* },
                    true,
                ),
            }
        };

        for (ident, dtype, param_axes) in &tile_in_params {
            let (offsets_expr, broadcast) = param_offsets(param_axes);
            // A broadcast operand has nothing to mask: it is one element,
            // and which lanes of the block are live is the *output's*
            // business, carried by its own tile's mask.
            //
            // Bare `None`/`Some`, not `::core::option::Option::None`: this
            // prelude is spliced into the kernel body, which is re-emitted
            // as device source and compiled by teenyc without a reachable
            // `::core`.
            let load_mask = if broadcast {
                quote! { None }
            } else {
                quote! { Some(in_bounds) }
            };
            let loaded = quote! {
                #hw_ident::load(
                    #ident.add_offsets(#offsets_expr),
                    #load_mask,
                    None,
                    &[],
                    None,
                    None,
                    None,
                    false,
                )
            };
            let (loaded, mask_tokens) = if broadcast {
                (
                    {
                        // Widen to the full tile shape: one extent per blocked
                        // axis, so a broadcast operand matches a K-D tile too.
                        // A block is a const generic's name or a decimal
                        // literal (teenygrad-1tl.9), so it is re-parsed into
                        // tokens rather than interpolated as a string -- a
                        // `String` in `quote!` would emit a string literal.
                        let shape: Vec<TokenStream2> = blocked_positions
                            .iter()
                            .map(|&i| {
                                axes[i]
                                    .block
                                    .as_ref()
                                    .expect("blocked axis carries a block")
                                    .parse()
                                    .expect("a block is an identifier or an integer literal")
                            })
                            .collect();
                        quote! { #hw_ident::broadcast_to(#loaded, &[#(#shape),*]) }
                    },
                    quote! { None },
                )
            } else {
                (loaded, quote! { Some(in_bounds) })
            };
            let load_stmt: syn::Stmt = syn::parse2(quote! {
                let #ident = Tile::<#hw_ident, #dtype> {
                    tensor: #loaded,
                    mask: #mask_tokens,
                };
            })
            .expect("generated tile load statement is valid Rust");
            stmts.push(load_stmt);
        }
        for (ident, dtype, param_axes) in &tile_out_params {
            let (offsets_expr, _) = param_offsets(param_axes);
            // `.add_offsets()` returns `HW::Tensor<HW::Pointer<D>>` (a tensor
            // of write addresses), not `HW::Tensor<D>` (a tensor of `D`
            // values) -- so the shadowed `Tile` is instantiated with
            // `HW::Pointer<D>` as its own dtype param, not `D` itself.
            let addr_stmt: syn::Stmt = syn::parse2(quote! {
                let #ident = Tile::<#hw_ident, #hw_ident::Pointer<#dtype>> {
                    tensor: #ident.add_offsets(#offsets_expr),
                    mask: Some(in_bounds),
                };
            })
            .expect("generated tile address statement is valid Rust");
            stmts.push(addr_stmt);
        }
        // teenygrad-y8aa: with `#[tile_loop(.., generate)]` the wrapper owns
        // the loop. The author's body IS one iteration, so it is spliced as the
        // loop body rather than appended, and the macro emits the carry
        // initialisation before it and the store after.
        //
        // That ordering is what retires constraint C4 of teenygrad-1nr.18.3's
        // analysis: Option A was rejected because wrapping an existing body
        // traps its trailing store inside the loop, but a generated store is
        // never in the body to begin with, so no marker is needed to find where
        // the loop ends. C3 holds by construction -- the loop is the wrapper.
        match &tile_loop {
            Some(l) if l.generate => {
                let (init_stmts, loop_stmt, store_stmt) = match generated_loop(
                    l,
                    &hw_ident,
                    &tile_in_params,
                    &tile_out_params,
                    &loop_scalars,
                    &loop_tiles,
                    &input,
                ) {
                    Ok(parts) => parts,
                    Err(e) => return e.to_compile_error().into(),
                };
                stmts.extend(init_stmts);
                stmts.push(loop_stmt);
                stmts.push(store_stmt);
            }
            _ => stmts.extend(input.block.stmts.iter().cloned()),
        }
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
    // rank) -- the same reason a flat elementwise spec has to be built
    // per node rather than shared as one `const`.
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
            let spec_axes = all_tile_param_attrs[0];
            let blocked = spec_axes
                .iter()
                .find(|a| a.block.is_some())
                .expect("checked above: exactly one axis carries `block = ..`");
            let block_str = blocked
                .block
                .as_ref()
                .expect("find() matched on is_some")
                .to_string();
            let extent_str = blocked.extent.to_string();
            let in_param_strs: Vec<String> = tile_in_params
                .iter()
                .map(|(id, _, _)| id.to_string())
                .collect();
            let out_param_strs: Vec<String> = tile_out_params
                .iter()
                .map(|(id, _, _)| id.to_string())
                .collect();
            let tile_spec = if spec_axes.len() > 1 {
                // teenygrad-1nr.18.1: with several declared axes the rank is no
                // longer a property of the graph node -- the signature states it --
                // so this emits a fixed-rank `tile_spec()` with one binding per
                // axis, the same shape the raw-pointer path already produces. A
                // single-axis kernel keeps the `tile_spec(rank)` form below, since
                // the same flat kernel really does apply at any rank.
                // teenygrad-1tl.4: each parameter reports *its own* declared
                // axes. Sharing one axis list across every tensor made a
                // broadcast operand claim axes it does not have -- a `(C,)`
                // bias against an `[N, C]` activation came back as rank 2 with
                // `N` blocked, so a scheduler would size its tile along an
                // axis that is not there. The prelude already honoured the
                // subset; only the spec was wrong.
                let tensor_spec =
                    |param: &str, attrs: &[TileAttrArgs]| -> Result<TokenStream2, syn::Error> {
                        let rank = attrs.len();
                        let reduction = match reduction_axis_of(attrs)? {
                            Some(i) => quote! { ::core::option::Option::Some(#i) },
                            None => quote! { ::core::option::Option::None },
                        };
                        let mut bindings: Vec<TokenStream2> = Vec::new();
                        let mut untiled: Vec<String> = Vec::new();
                        for (i, axis) in attrs.iter().enumerate() {
                            match &axis.block {
                                Some(block) => {
                                    let block_s = block.to_string();
                                    let extent_s = axis.extent.to_string();
                                    let window = window_tokens(axis);
                                    bindings.push(quote! {
                                        ::teeny_core::model::TileAxisBinding {
                                            dims: &[#i],
                                            block_const: #block_s,
                                            extent_param: #extent_s,
                                            window: #window,
                                            divide_by: ::core::option::Option::None,
                                        }
                                    });
                                }
                                // A window on an unblocked axis is a binding
                                // with the literal block 1, exactly as on the
                                // raw-pointer path below. teenygrad-1tl.7 added
                                // it there and missed this copy, so the same
                                // attribute silently meant different things
                                // depending on whether a kernel took
                                // `In<Tile<..>>` or a tagged pointer. No
                                // route-1 kernel declares a window today --
                                // they are the flat elementwise family -- so
                                // nothing was mis-specified, but the asymmetry
                                // was a trap.
                                None if axis.window.is_some() => {
                                    let extent_s = axis.extent.to_string();
                                    let window = window_tokens(axis);
                                    bindings.push(quote! {
                                        ::teeny_core::model::TileAxisBinding {
                                            dims: &[#i],
                                            block_const: "1",
                                            extent_param: #extent_s,
                                            window: #window,
                                            divide_by: ::core::option::Option::None,
                                        }
                                    });
                                }
                                None => untiled.push(
                                    axis.name
                                        .as_ref()
                                        .map(syn::LitStr::value)
                                        .unwrap_or_else(|| axis.extent.to_string()),
                                ),
                            }
                        }
                        Ok(quote! {
                            ::teeny_core::model::TensorTileSpec {
                                param: #param,
                                rank: #rank,
                                axes: &[ #(#bindings),* ],
                                reduction_axis: #reduction,
                                untiled_dims: &[ #(#untiled),* ],
                            }
                        })
                    };
                let specs: Result<(Vec<TokenStream2>, Vec<TokenStream2>), syn::Error> = (|| {
                    let ins = tile_in_params
                        .iter()
                        .map(|(id, _, attrs)| tensor_spec(&id.to_string(), attrs))
                        .collect::<Result<Vec<_>, _>>()?;
                    let outs = tile_out_params
                        .iter()
                        .map(|(id, _, attrs)| tensor_spec(&id.to_string(), attrs))
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok((ins, outs))
                })(
                );
                let (input_specs, output_specs) = match specs {
                    Ok(pair) => pair,
                    Err(e) => return e.to_compile_error().into(),
                };
                let tile_spec_tokens = quote! {
                    /// Declarative tile-shape metadata derived from this kernel's
                    /// `#[tile(...)]`-tagged `In<Tile<..>>`/`Out<Tile<..>>`
                    /// parameters (teenygrad-1nr.18.1). Fixed rank: the signature
                    /// declares every axis.
                    pub fn tile_spec() -> ::teeny_core::model::KernelTileSpec {
                        const INPUTS: &[::teeny_core::model::TensorTileSpec] =
                            &[ #(#input_specs),* ];
                        const OUTPUTS: &[::teeny_core::model::TensorTileSpec] =
                            &[ #(#output_specs),* ];
                        ::teeny_core::model::KernelTileSpec {
                            inputs: INPUTS,
                            outputs: OUTPUTS,
                            loop_spec: #loop_spec_tokens,
                        }
                    }
                };
                tile_spec_tokens
            } else {
                let window = window_tokens(blocked);
                let tile_spec_tokens = quote! {
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
                                    window: #window,
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
                            loop_spec: #loop_spec_tokens,
                        }
                    }
                };
                tile_spec_tokens
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
                            let window = window_tokens(axis);
                            tiled_axis_tokens.push(quote! {
                                ::teeny_core::model::TileAxisBinding {
                                    dims: &[#i],
                                    block_const: #block_str,
                                    extent_param: #extent_str,
                                    window: #window,
                                    divide_by: ::core::option::Option::None,
                                }
                            });
                        }
                        // An axis with a window but no block is windowed with a
                        // *fixed* block of 1: one program instance covers one
                        // element of it. conv2d blocks `OW` alone, so its `pid`
                        // decode yields a scalar `oh` and a `BLOCK_OW`-wide
                        // `ow` range -- yet `x_ptr`'s H axis is still read
                        // through a `KH`-tall sliding window, and dropping that
                        // window made the input look as though it were read at
                        // full extent (teenygrad-1tl.7).
                        //
                        // So it becomes a real binding carrying its window,
                        // with `block_const: "1"` -- a decimal literal, the
                        // same spelling `shape_consts` already accepts for a
                        // scalar accumulator, because there is no `BLOCK_OH`
                        // const to name. `resolve_inputs` then derives the
                        // receptive field `(1 - 1) * STRIDE_H + KH = KH`:
                        // exactly the rows one output row reads.
                        //
                        // This describes the body as written; it does not make
                        // the axis tileable. Letting a consumer *choose* a
                        // block for H means rewriting the body's `pid` decode,
                        // which is a port, not a declaration
                        // (teenygrad-1nr.18.4).
                        None if axis.window.is_some() => {
                            let extent_str = axis.extent.to_string();
                            let window = window_tokens(axis);
                            tiled_axis_tokens.push(quote! {
                                ::teeny_core::model::TileAxisBinding {
                                    dims: &[#i],
                                    block_const: "1",
                                    extent_param: #extent_str,
                                    window: #window,
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
                let reduction = match reduction_axis_of(axes) {
                    Ok(Some(i)) => quote! { ::core::option::Option::Some(#i) },
                    Ok(None) => quote! { ::core::option::Option::None },
                    Err(e) => return e.to_compile_error().into(),
                };
                let tensor_spec = quote! {
                    ::teeny_core::model::TensorTileSpec {
                        param: #param_str,
                        rank: #rank,
                        axes: &[ #(#tiled_axis_tokens),* ],
                        reduction_axis: #reduction,
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
                        loop_spec: #loop_spec_tokens,
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
                // A swizzled `pid` decode has no `GridSpec` that could describe
                // it: `GridAxisBinding::dim` documents multiple axes on one dim
                // as a mixed-radix decode, and the matmul family's `GROUP_M`
                // grouping is not one. Emitting axes anyway would assert a
                // decode the body does not perform, and a fused rider reads the
                // anchor's decoded values, so it would hand out wrong indices
                // (teenygrad-1tl.10).
                Some(_) if matches!(grid_order, Some(TileGridArgs::Swizzled)) => quote! {},
                Some((_, axes)) => {
                    // `#[tile_grid(order = [..])]` states the body's real
                    // decode order when it differs from the output's dim
                    // order; without it, dim order *is* the decode order.
                    let ordered: Vec<&TileAttrArgs> = match &grid_order {
                        None => axes.iter().collect(),
                        // Unreachable: handled before this match, which returns
                        // an empty `grid_spec` for a swizzled kernel.
                        Some(TileGridArgs::Swizzled) => axes.iter().collect(),
                        Some(TileGridArgs::Order(order)) => {
                            let mut picked = Vec::new();
                            for want in order {
                                let want_s = want.to_string();
                                let Some(found) = axes.iter().find(|a| {
                                    a.name
                                        .as_ref()
                                        .map(syn::LitStr::value)
                                        .unwrap_or_else(|| a.extent.to_string())
                                        == want_s
                                }) else {
                                    return syn::Error::new_spanned(
                                        want,
                                        format!(
                                            "`#[tile_grid(order = ..)]` names `{want_s}`, which \
                                             is not an axis of this kernel's output parameter"
                                        ),
                                    )
                                    .to_compile_error()
                                    .into();
                                };
                                picked.push(found);
                            }
                            // A subset is meaningful, not an error: the list is
                            // *the grid axes*, and an axis the body covers with
                            // a loop is not one. `batch_norm_normalize_forward`
                            // runs one program per channel and walks N with a
                            // `while` loop, so its grid is `[C]` even though
                            // its output is `[N, C]` (teenygrad-1tl.8).
                            //
                            // A repeat is still a bug -- it would duplicate a
                            // grid axis -- and so is naming an axis the output
                            // does not have, checked above.
                            let mut seen: Vec<String> = Vec::new();
                            for want in order {
                                let w = want.to_string();
                                if seen.contains(&w) {
                                    return syn::Error::new_spanned(
                                        want,
                                        format!("`#[tile_grid(order = ..)]` names `{w}` twice"),
                                    )
                                    .to_compile_error()
                                    .into();
                                }
                                seen.push(w);
                            }
                            picked
                        }
                    };
                    let axis_tokens: Vec<TokenStream2> = ordered
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
        } else if tile_loop.is_some() {
            // A kernel that declares a loop but tags no parameter still gets a
            // `tile_spec()`, carrying the loop and nothing else.
            //
            // `flash_attention2_forward` is the case: every one of its tensors
            // has `BH` as dim 0, and `BH` is the *grid* extent of `Axis::Y`,
            // never a kernel parameter -- the body only ever needs `pid_bh`.
            // `#[tile(extent = ..)]` names a parameter, so no axis of any of
            // its tensors can be named truthfully, and inventing a `bh: i32`
            // argument the body never reads would change a device kernel's ABI
            // to carry metadata. So its pointers stay untagged, which is the
            // shape the pre-revival design had too (teenygrad-1tl.11).
            //
            // Without this arm the loop attributes were silently ignored: they
            // parsed, compiled, and produced nothing, because `loop_spec` rides
            // inside `tile_spec()` and `tile_spec()` was only generated when a
            // parameter was tagged.
            //
            // `resolve_inputs` rejects a spec with no outputs, which is the
            // right answer rather than a problem: there is no output tile to
            // resolve against, and this kernel is not reachable from the graph
            // at all yet (no `Op` variant, teenygrad-1nr.32).
            (
                quote! {
                    /// Declarative tile metadata for a kernel that declares an
                    /// accumulation loop but tags no parameter: `loop_spec`
                    /// only (teenygrad-1tl.11).
                    pub fn tile_spec() -> ::teeny_core::model::KernelTileSpec {
                        ::teeny_core::model::KernelTileSpec {
                            inputs: &[],
                            outputs: &[],
                            loop_spec: #loop_spec_tokens,
                        }
                    }
                },
                quote! {},
            )
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
