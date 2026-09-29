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

//! Guards the one rule every rung of `teenygrad-1tl` has to follow: a forward
//! kernel that takes `Tile` parameters must say what its axes are.
//!
//! ## Why this has to read the source
//!
//! The failure it catches is invisible at the type level. `#[tiled_kernel]`
//! generates `tile_spec()` *only* when every `Tile` parameter carries
//! `#[tile(...)]`; with no attributes it silently falls back to the implicit
//! `BLOCK_SIZE`/`n_elements` convention, the kernel compiles, its numeric
//! tests pass, and `tile_spec()` is simply absent. There is no method to call
//! and therefore nothing a normal test could assert against -- the only way to
//! see the omission is to look at the declaration. So this parses the crate's
//! own `src/` with `syn` and checks the signatures.
//!
//! ## What counts as in scope
//!
//! Keying off `#[tiled_kernel]` alone would be wrong: `matmul_forward` and
//! `flash_attention2_forward` carry it purely for dtype dispatch, with plain
//! `In<T::Pointer<D>>` parameters and no tiling meaning at all. Flagging them
//! would report a non-defect.
//!
//! The rule is therefore scoped to a **forward** kernel with at least one
//! `In<Tile<..>>`/`Out<Tile<..>>` parameter. Backward kernels are out of this
//! epic's scope (`elu_backward` and `selu_backward` have `Tile` parameters and
//! are deliberately left alone), and a kernel still on raw pointers is merely
//! unconverted, which is the normal state until its own rung lands.

use std::path::{Path, PathBuf};

use syn::{Item, ItemFn, PatType, Type};

/// A forward kernel that takes `Tile` parameters but does not declare them.
#[derive(Debug)]
struct Undeclared {
    kernel: String,
    file: PathBuf,
    /// Names of the `Tile`-typed parameters carrying no `#[tile(...)]`.
    params: Vec<String>,
}

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("read dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The last path segment's identifier, for a plain `Type::Path`.
fn last_segment(ty: &Type) -> Option<&syn::PathSegment> {
    match ty {
        Type::Path(tp) => tp.path.segments.last(),
        _ => None,
    }
}

/// `true` for `In<Tile<..>>` / `Out<Tile<..>>`; `false` for a raw
/// `In<T::Pointer<D>>`. Matched on the syntax tree rather than on a printed
/// string, so it does not depend on spacing or on how the path is spelled.
fn is_tile_param(ty: &Type) -> bool {
    let Some(seg) = last_segment(ty) else {
        return false;
    };
    if seg.ident != "In" && seg.ident != "Out" {
        return false;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return false;
    };
    args.args.iter().any(|arg| {
        matches!(arg, syn::GenericArgument::Type(inner)
            if last_segment(inner).is_some_and(|s| s.ident == "Tile"))
    })
}

/// A parameter's binding name, for the failure message.
fn param_name(pt: &PatType) -> String {
    match &*pt.pat {
        syn::Pat::Ident(id) => id.ident.to_string(),
        _ => "<pattern>".to_string(),
    }
}

/// Collects every `#[tiled_kernel]`/`#[tiled_kernel(..)]` function in one file,
/// including those nested in inline modules.
fn tiled_kernels(items: &[Item], out: &mut Vec<ItemFn>) {
    for item in items {
        match item {
            Item::Fn(f) if f.attrs.iter().any(|a| a.path().is_ident("tiled_kernel")) => {
                out.push(f.clone());
            }
            Item::Mod(m) => {
                if let Some((_, inner)) = &m.content {
                    tiled_kernels(inner, out);
                }
            }
            _ => {}
        }
    }
}

#[test]
fn test_every_tile_param_forward_kernel_declares_its_axes() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = Path::new(manifest).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(
        files.len() > 50,
        "only found {} source files under {} -- the walk is probably wrong, and a test that \
         silently scans nothing would pass forever",
        files.len(),
        src.display()
    );

    let mut checked = 0usize;
    let mut offenders: Vec<Undeclared> = Vec::new();

    for file in &files {
        let text = std::fs::read_to_string(file).expect("read source file");
        // A file this crate cannot parse is a separate problem; do not let it
        // masquerade as a passing declaration check.
        let parsed = syn::parse_file(&text)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", file.display()));

        let mut kernels = Vec::new();
        tiled_kernels(&parsed.items, &mut kernels);

        for f in kernels {
            let name = f.sig.ident.to_string();
            // Backward kernels are out of scope for teenygrad-1tl.
            if name.contains("backward") {
                continue;
            }
            let tile_params: Vec<&PatType> = f
                .sig
                .inputs
                .iter()
                .filter_map(|arg| match arg {
                    syn::FnArg::Typed(pt) => Some(pt),
                    syn::FnArg::Receiver(_) => None,
                })
                .filter(|pt| is_tile_param(&pt.ty))
                .collect();

            // No `Tile` parameters: either dtype-dispatch only (matmul, flash
            // attention) or not yet converted. Neither is a defect.
            if tile_params.is_empty() {
                continue;
            }
            checked += 1;

            let missing: Vec<String> = tile_params
                .iter()
                .filter(|pt| !pt.attrs.iter().any(|a| a.path().is_ident("tile")))
                .map(|pt| param_name(pt))
                .collect();
            if !missing.is_empty() {
                offenders.push(Undeclared {
                    kernel: name,
                    file: file.clone(),
                    params: missing,
                });
            }
        }
    }

    assert!(
        checked >= 8,
        "only {checked} forward kernels with Tile parameters were found; the scan is too narrow \
         to be enforcing anything"
    );

    if !offenders.is_empty() {
        let mut report = String::new();
        for o in &offenders {
            report.push_str(&format!(
                "\n  {} ({}): undeclared -- {}",
                o.kernel,
                o.file.strip_prefix(manifest).unwrap_or(&o.file).display(),
                o.params.join(", "),
            ));
        }
        panic!(
            "{} forward kernel(s) take Tile parameters without declaring their tile axes.{}\n\n\
             Add `#[tile(block = .., extent = ..)]` to every Tile parameter. Without it the \
             kernel silently falls back to the implicit BLOCK_SIZE/n_elements convention and \
             generates no tile_spec(), so TileGraph::propagate treats it as a hard boundary. \
             See contributing/TiledKernelRecipe.md.",
            offenders.len(),
            report
        );
    }
}

/// The source scan above proves a declaration *exists*; this proves it means
/// something. Each backfilled kernel's generated `tile_spec()` must describe one
/// flat axis shared by `x` and `y` and must `validate()`.
///
/// These six take the `tile_spec(rank)` form rather than a fixed-rank one: a
/// single flat axis really does apply at any rank, so the rank is a property of
/// the graph node, not of the signature. (Contrast the multi-axis kernels, whose
/// signature states the rank -- see `nn::tensor`'s own test module.)
#[test]
fn test_the_backfilled_kernels_generate_validating_specs() {
    use teeny_kernels::nn::activation::{
        elu::{EluForward, SeluForward},
        log_sigmoid::LogSigmoidForward,
        sigmoid::SigmoidForward,
        tanh::TanhForward,
    };
    use teeny_kernels::nn::tensor::elemwise_unary::ElemwiseExpForward;

    macro_rules! check {
        ($($kernel:ty),+ $(,)?) => {
            $({
                let name = stringify!($kernel);
                for rank in 1..=4usize {
                    let spec = <$kernel>::tile_spec(rank);
                    assert_eq!(spec.loop_spec, None, "{name}: flat kernels carry no loop");
                    assert_eq!(
                        (spec.inputs.len(), spec.outputs.len()),
                        (1, 1),
                        "{name}: one input, one output"
                    );
                    assert_eq!(
                        (spec.inputs[0].param, spec.outputs[0].param),
                        ("x", "y"),
                        "{name}: params are named from the signature"
                    );
                    for tensor in [spec.inputs[0], spec.outputs[0]] {
                        assert_eq!(tensor.rank, rank, "{name}: rank comes from the node");
                        assert_eq!(tensor.axes.len(), 1, "{name}: one flattened axis");
                        assert_eq!(
                            tensor.axes[0].dims,
                            (0..rank).collect::<Vec<_>>(),
                            "{name}: the axis spans every dim"
                        );
                        assert_eq!(tensor.axes[0].block_const, "BLOCK_SIZE", "{name}");
                        assert_eq!(tensor.axes[0].extent_param, "n_elements", "{name}");
                        assert_eq!(tensor.axes[0].window, None, "{name}: no window yet");
                        assert_eq!(tensor.axes[0].divide_by, None, "{name}");
                        assert_eq!(tensor.reduction_axis, None, "{name}");
                    }
                    spec.validate()
                        .unwrap_or_else(|e| panic!("{name}: derived spec must validate: {e}"));
                }
            })+
        };
    }

    check!(
        EluForward<f32>,
        SeluForward<f32>,
        SigmoidForward<f32>,
        LogSigmoidForward<f32>,
        TanhForward<f32>,
        ElemwiseExpForward<f32>,
    );
}
