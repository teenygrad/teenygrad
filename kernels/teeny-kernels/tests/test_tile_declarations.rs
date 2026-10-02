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
    /// Names of the `Tile`-typed parameters carrying none of `#[tile(...)]`,
    /// `#[tile_loop_scalar(...)]` or `#[tile_loop_tile(...)]`.
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
                // `#[tile_loop_scalar]` counts as a declaration too. Such an operand is
                // *indexed*, not tiled -- it is read one element per loop iteration at
                // a loop-dependent offset and broadcast -- so it has no axes to declare
                // and requiring `#[tile(block = .., extent = ..)]` of it would mean
                // writing something false (teenygrad-y8aa).
                .filter(|pt| {
                    !pt.attrs.iter().any(|a| {
                        a.path().is_ident("tile")
                            || a.path().is_ident("tile_loop_scalar")
                            || a.path().is_ident("tile_loop_tile")
                    })
                })
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

/// The epic's closing audit: every in-scope forward kernel is in one of three
/// buckets, and none is unaccounted for (teenygrad-1tl.12).
///
/// The buckets are route 1 (a full `In<Tile<..>>` conversion), route 2
/// (metadata on tagged raw pointers), and a recorded reason it has no spec.
/// Loss kernels are out of scope: the epic covers inference kernels for now,
/// recorded on `teenygrad-1tl`.
///
/// This is a test rather than a one-off count so the answer cannot rot. Writing
/// it as a script first was instructive -- it misclassified three kernels in
/// three different ways before it was right. A `#[tiled_kernel(backward = ..)]`
/// carries arguments, so an exact-bracket match misses it; a multi-line
/// `#[tile_loop(..)]` puts bare `generate` on its own line, which breaks a naive
/// backward walk; and two of the recorded reasons *quote* `#[tiled_kernel]` in
/// prose, so matching the raw text of the preceding block counts a comment as a
/// declaration. Hence: attributes are read from the syntax tree, and reasons
/// only from doc comments.
#[test]
fn test_the_closing_audit_leaves_no_forward_kernel_unaccounted_for() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let src = Path::new(manifest).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);

    let mut route1 = Vec::new();
    let mut route2 = Vec::new();
    let mut reasoned = Vec::new();
    let mut unaccounted = Vec::new();
    let mut losses = Vec::new();

    for file in &files {
        let text = std::fs::read_to_string(file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        let parsed = syn::parse_file(&text)
            .unwrap_or_else(|e| panic!("failed to parse {}: {e}", file.display()));

        let in_loss = file.components().any(|c| c.as_os_str() == "loss");

        // Declared kernels, from the syntax tree.
        let mut declared = Vec::new();
        tiled_kernels(&parsed.items, &mut declared);
        for f in &declared {
            let name = f.sig.ident.to_string();
            if !name.ends_with("_forward") {
                continue;
            }
            let any_tile_param = f.sig.inputs.iter().any(|arg| match arg {
                syn::FnArg::Typed(pt) => is_tile_param(&pt.ty),
                _ => false,
            });
            if any_tile_param {
                route1.push(name);
            } else {
                route2.push(name);
            }
        }

        // Undeclared forwards: a recorded reason, or unaccounted. The reason is
        // read from doc comments only -- never from the text of the block, since
        // two reasons quote the attribute they are explaining.
        for item in &parsed.items {
            let syn::Item::Fn(f) = item else { continue };
            let name = f.sig.ident.to_string();
            if !name.ends_with("_forward") {
                continue;
            }
            if f.attrs.iter().any(|a| a.path().is_ident("tiled_kernel")) {
                continue;
            }
            if in_loss {
                losses.push(name);
                continue;
            }
            // Read from the file's text, not the syntax tree: the recorded
            // reasons are `//` comments, which `syn` does not keep. Only `//`
            // lines are taken -- two of the reasons quote `#[tiled_kernel]` in
            // prose, so scanning the whole preceding block would read a comment
            // as a declaration.
            let comments = leading_line_comments(&text, &name);
            let has_reason = [
                "teenygrad-1tl",
                "teenygrad-12l6",
                "deliberately",
                "undeclared",
            ]
            .iter()
            .any(|needle| comments.contains(needle));
            if has_reason {
                reasoned.push(name);
            } else {
                unaccounted.push(format!("{name} ({})", file.display()));
            }
        }
    }

    assert!(
        unaccounted.is_empty(),
        "every in-scope forward kernel must be declared or carry a recorded \
         reason; these are neither:\n  {}",
        unaccounted.join("\n  ")
    );

    // The epic's opening table, restated. Guard rails rather than exact counts:
    // a new kernel should not have to touch this test, but a collapse in either
    // bucket should be noticed.
    assert!(
        route1.len() + route2.len() >= 125,
        "declared forwards fell to {} (route 1 {}, route 2 {})",
        route1.len() + route2.len(),
        route1.len(),
        route2.len()
    );
    assert!(
        !route1.is_empty() && !route2.is_empty(),
        "both routes should still be in use: route 1 {}, route 2 {}",
        route1.len(),
        route2.len()
    );
    assert!(
        losses.len() >= 9,
        "the nine undeclared loss kernels are excluded by scope, found {}",
        losses.len()
    );
}

/// The `//` comment lines immediately above `pub fn {name}<` in `text`.
///
/// Walks back over comments and attributes and stops at the first line that is
/// neither -- a blank line, a closing brace, or another item. Returns only the
/// comment lines, so an attribute quoted inside prose is not mistaken for the
/// attribute itself.
fn leading_line_comments(text: &str, name: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let needle = format!("pub fn {name}<");
    let Some(at) = lines
        .iter()
        .position(|l| l.trim_start().starts_with(&needle))
    else {
        return String::new();
    };
    let mut out = Vec::new();
    for l in lines[..at].iter().rev() {
        let t = l.trim_start();
        if t.starts_with("//") {
            out.push(t);
        } else if t.starts_with('#') || t.starts_with(')') || t.contains('=') {
            continue;
        } else {
            break;
        }
    }
    out.join("\n")
}
