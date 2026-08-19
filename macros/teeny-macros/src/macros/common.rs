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

//! Helpers shared by [`super::kernel`] and [`super::tiled_kernel`]: signature
//! parsing, pointer-marker classification, and `#[kernel(...)]`/
//! `#[tiled_kernel(...)]` attribute parsing. Both macros accept the same
//! top-level attribute syntax and pointer-marker conventions.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Expr, GenericArgument, Ident, MetaNameValue, Pat, PathArguments, Token, Type, parse::Parser,
    punctuated::Punctuated,
};

pub(crate) fn to_pascal_case(s: &str) -> String {
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

/// Classification of a kernel pointer parameter for [`KernelIo`] / ABI wrapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PtrArgKind {
    In,
    Out,
    InOut,
    /// Bare `HW::Pointer<D>` with no In/Out marker.
    Raw,
}

pub(crate) fn extract_single_generic_type(seg: &syn::PathSegment) -> Option<Type> {
    if let PathArguments::AngleBracketed(ab) = &seg.arguments
        && ab.args.len() == 1
        && let GenericArgument::Type(inner) = &ab.args[0]
    {
        return Some(inner.clone());
    }
    None
}

/// If `ty` is `HW_IDENT::Pointer<Inner>` (two-segment path), return `Inner`.
pub(crate) fn extract_hw_pointer_dtype(ty: &Type, hw_ident: &Ident) -> Option<Type> {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
    {
        let segs = &tp.path.segments;
        if segs.len() == 2
            && segs[0].ident == *hw_ident
            && segs[1].ident == "Pointer"
            && let PathArguments::AngleBracketed(ab) = &segs[1].arguments
            && ab.args.len() == 1
            && let GenericArgument::Type(inner) = &ab.args[0]
        {
            return Some(inner.clone());
        }
    }
    None
}

/// Classify `In<HW::Pointer<D>>` / `Out<…>` / `InOut<…>` / bare `HW::Pointer<D>`.
///
/// Returns `(kind, dtype)` where `dtype` is the `D` in `Pointer<D>`.
pub(crate) fn classify_pointer_arg(ty: &Type, hw_ident: &Ident) -> Option<(PtrArgKind, Type)> {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && let Some(last) = tp.path.segments.last()
    {
        let kind = match last.ident.to_string().as_str() {
            "In" => Some(PtrArgKind::In),
            "Out" => Some(PtrArgKind::Out),
            "InOut" => Some(PtrArgKind::InOut),
            _ => None,
        };
        if let Some(kind) = kind {
            let wrapped = extract_single_generic_type(last)?;
            let dtype = extract_hw_pointer_dtype(&wrapped, hw_ident)?;
            return Some((kind, dtype));
        }
    }
    extract_hw_pointer_dtype(ty, hw_ident).map(|dtype| (PtrArgKind::Raw, dtype))
}

/// If `ty` is a (possibly marked) pointer arg, return its element dtype.
pub(crate) fn extract_pointer_inner(ty: &Type, hw_ident: &Ident) -> Option<Type> {
    classify_pointer_arg(ty, hw_ident).map(|(_, dtype)| dtype)
}

/// Strip `In` / `Out` / `InOut` wrappers for the device-side source string.
///
/// Markers are host-only metadata for [`KernelIo`]; the MLIR backend only knows
/// about bare `T::Pointer<D>` / `LlvmPointer` and panics on unmarked ADTs.
pub(crate) fn unwrap_pointer_marker(ty: &Type) -> Type {
    if let Type::Path(tp) = ty
        && tp.qself.is_none()
        && let Some(last) = tp.path.segments.last()
    {
        match last.ident.to_string().as_str() {
            "In" | "Out" | "InOut" => {
                if let Some(inner) = extract_single_generic_type(last) {
                    return inner;
                }
            }
            _ => {}
        }
    }
    ty.clone()
}

/// Extract the ident from a bare single-segment type, e.g. `D` → `Some(D)`.
pub(crate) fn simple_type_ident(ty: &Type) -> Option<Ident> {
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

pub(crate) fn pat_to_str(pat: &Pat) -> String {
    quote!(#pat).to_string()
}

/// The set of concrete scalar dtypes permitted by a type-parameter trait bound.
///
/// Used when a kernel opts into dispatch (via `dtypes`/`backward`) but omits an
/// explicit `dtypes` list: "no dtypes specified" means "every dtype the bound
/// allows". Only dtypes with concrete Rust impls are included (e.g. `f16`/`bf16`
/// are marker-only and cannot be monomorphized). Returns `None` for an unknown
/// bound.
pub(crate) fn all_dtypes_for_bound(bound: &str) -> Option<&'static [&'static str]> {
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
pub(crate) fn dtype_ident_to_repr(id: &Ident) -> Option<TokenStream2> {
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

/// Parsed `#[kernel(...)]` / `#[tiled_kernel(...)]` attribute arguments.
#[derive(Default)]
pub(crate) struct KernelAttrs {
    /// Declared supported dtypes (idents such as `f32`, `f64`, `i32`).
    pub(crate) dtypes: Vec<Ident>,
    /// Optional paired backward kernel struct ident.
    pub(crate) backward: Option<Ident>,
}

/// Parse the attribute tokens of `#[kernel(dtypes = [..], backward = Foo)]`
/// (also used verbatim by `#[tiled_kernel(...)]`).
pub(crate) fn parse_kernel_attrs(attrs: TokenStream) -> Result<KernelAttrs, syn::Error> {
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
