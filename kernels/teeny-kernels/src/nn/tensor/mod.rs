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

pub mod channel_bias_add;
pub mod channel_cat;
pub mod channel_chunk;
pub mod elemwise_add;
pub mod elemwise_binary;
pub mod elemwise_unary;
pub mod reduction;
pub mod transpose;
pub mod upsample_nearest2d;

#[cfg(test)]
mod tests {
    //! Probe for the N-axis auto-prelude (teenygrad-1nr.18.1).
    //!
    //! Every real multi-axis kernel in the tree also loops -- conv and pool
    //! over a receptive field, the norms over a row, `channel_bias_add` over
    //! its own tile range -- so none of them can exercise the multi-axis
    //! prelude on its own until `teenygrad-1nr.18.3` lands the wrapper loop.
    //! The kernel below exists so the generated indexing is type-checked and
    //! its spec asserted in the meantime. It is defined here rather than in
    //! `src/` because nothing outside this test needs it: it is not lowered
    //! from any `Op` and is not part of the public kernel set.
    #![allow(non_snake_case)]

    use teeny_core::dtype::Num;
    use teeny_macros::tiled_kernel;
    use teeny_triton::triton::{
        types::{AddOffsets, Comparison},
        *,
    };

    /// Copy, over a rank-2 `[H, W]` tensor: `H` is one index per CTA, `W` is
    /// block-tiled. Exercises the flat-pid decode and the row-major stride
    /// derivation the prelude generates.
    #[tiled_kernel]
    pub fn multi_axis_probe_forward<T: Triton, D: Num, const BLOCK_W: i32>(
        #[tile(extent = H)]
        #[tile(block = BLOCK_W, extent = W)]
        x: In<Tile<T, D>>,
        #[tile(extent = H)]
        #[tile(block = BLOCK_W, extent = W)]
        y: Out<Tile<T, D>>,
        H: i32,
        W: i32,
    ) where
        T::I32Tensor: types::Tensor<i32, 1>,
        T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
        T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    {
        T::store(y.tensor, x.tensor, x.mask, &[], None, None);
    }

    /// Feasibility probe for the shape teenygrad-y8aa will generate: a carry
    /// initialised before a loop, threaded through it by assignment, and stored
    /// once afterwards.
    ///
    /// This is the *inline* reading of Option C from teenygrad-1nr.18.3's
    /// design analysis -- the wrapper owns the loop and the author's body is
    /// spliced in as the loop body -- rather than the literal reading, where
    /// the iteration is its own function taking and returning the carry. The
    /// literal reading was measured and does not work, for two stacked reasons:
    ///
    /// 1. `#[tiled_kernel]` emits only the kernel function into the device
    ///    source, so a helper in the same module is not there at all --
    ///    `error[E0425]: cannot find function ... in this scope`, from the
    ///    generated source rather than from rustc on the host.
    /// 2. Even once emitted, `teenyc-3af.4` applies: an ordinary generic
    ///    function whose signature involves a tensor type is treated as an
    ///    intrinsic stub and its body skipped, producing a dangling `tt.call`
    ///    that ICEs at MLIR verification. That fix exists on teenyc's
    ///    `feat/tile-layout-struct-support` and is not in the installed
    ///    compiler.
    ///
    /// The inline shape needs neither: it is what every hand-written
    /// accumulating kernel in the tree already compiles to, `conv2d_forward`
    /// included. This probe pins that down so the choice is not re-litigated.
    #[tiled_kernel]
    #[tile_loop(trip_count = [trips], count = trips, generate)]
    #[tile_carry(acc = [BLOCK_N])]
    pub fn carry_loop_probe_forward<T: Triton, D: Num, const BLOCK_N: i32>(
        #[tile(block = BLOCK_N, extent = n_elements)] x: In<Tile<T, D>>,
        #[tile(block = BLOCK_N, extent = n_elements)] y: Out<Tile<T, D>>,
        n_elements: i32,
        trips: i32,
    ) where
        T::I32Tensor: types::Tensor<i32, 1>,
        T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
        T::Tensor<D>: core::ops::Add<T::Tensor<D>, Output = T::Tensor<D>>,
        T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    {
        // One iteration, and nothing else. The carry's initialisation, the
        // loop around this, and the store after it are all generated
        // (teenygrad-y8aa).
        acc = acc + x.tensor;
    }

    /// Compiles the probe with the real teenyc, which is the point: a
    /// type-check alone never reaches `codegen_function` (teenygrad-y8aa).
    #[test]
    fn test_carry_threaded_through_a_loop_compiles() {
        let kernel = CarryLoopProbeForward::<f32>::new(128);
        let target = teeny_runtime::reference_target();
        let compiled = teeny_runtime::compile_kernel(&kernel, &target, true, false)
            .expect("a carry threaded through a loop must compile");
        assert!(!compiled.is_empty(), "compile produced no artifact");
    }

    /// The three generated pieces are present, and in the order that makes the
    /// loop the wrapper's (teenygrad-y8aa).
    ///
    /// Order is the whole contract: the carry is initialised *before* the loop
    /// and stored *after* it, so the author's body is one iteration and nothing
    /// else. Getting the store inside the loop is precisely what sank Option A.
    #[test]
    fn test_the_generated_loop_initialises_before_and_stores_after() {
        let src = CarryLoopProbeForward::<f32>::new(128).source;

        let init = src
            .find("let mut acc")
            .expect("the carry's initialisation is generated");
        let loop_at = src
            .find("for __tile_loop_idx in 0 .. (trips)")
            .expect("the loop is generated, with the evaluable `count` as its bound");
        let store = src
            .find("store(y.tensor, acc,")
            .expect("the store of the carry is generated");

        assert!(
            init < loop_at,
            "the carry must be initialised before the loop"
        );
        assert!(loop_at < store, "the carry must be stored after the loop");
        assert!(
            !src[..loop_at].contains("store(y.tensor"),
            "nothing may store before the loop"
        );

        // The author's one statement is inside the loop, not after it.
        let body = src
            .find("acc = acc + x.tensor")
            .expect("the author's body is spliced into the loop");
        assert!(
            loop_at < body && body < store,
            "the body belongs between the loop header and the store"
        );
    }

    /// `generate` does not disturb the metadata teenygrad-1nr.18.3 delivered:
    /// `trip_count` still reports the declared *names*, not the evaluable
    /// `count` expression, and the carry still reports its shape.
    #[test]
    fn test_generate_leaves_the_declared_loop_spec_intact() {
        // Rank 1: the probe declares one axis. A route-1 spec is built per
        // node, so the rank is an argument rather than part of the signature.
        let spec = CarryLoopProbeForward::<f32>::tile_spec(1);
        let l = spec.loop_spec.expect("the loop is declared");
        assert_eq!(l.trip_count_factors, &["trips"]);
        let carries: Vec<(&str, &[&str])> =
            l.carries.iter().map(|c| (c.name, c.shape_consts)).collect();
        assert_eq!(carries, vec![("acc", &["BLOCK_N"][..])]);
    }

    /// Probe for the generated loop-index decode -- Option D of
    /// teenygrad-1nr.18.3's analysis (teenygrad-y8aa).
    ///
    /// Three loop axes, shaped like conv2d's `(C_IN/G, KH, KW)`, so the
    /// generated arithmetic can be compared against the decode conv2d writes by
    /// hand:
    ///
    /// ```text
    /// kw = idx % KW;  kh = idx / KW % KH;  c_in_local = idx / (KW * KH)
    /// ```
    ///
    /// The extents are deliberately expressions, not bare consts: conv2d's
    /// outermost extent is `(C_IN / G)`, which is why an axis takes an
    /// expression and why the product of the extents can stand in for `count`.
    #[tiled_kernel]
    #[tile_loop(
        trip_count = [C_IN, G, KH, KW],
        axes = [c_in_local = (C_IN / G), kh = KH, kw = KW],
        generate
    )]
    #[tile_carry(acc = [BLOCK_N])]
    #[allow(unused_variables)]
    pub fn loop_decode_probe_forward<T: Triton, D: Num, const BLOCK_N: i32>(
        #[tile(block = BLOCK_N, extent = n_elements)] x: In<Tile<T, D>>,
        #[tile(block = BLOCK_N, extent = n_elements)] y: Out<Tile<T, D>>,
        n_elements: i32,
        C_IN: i32,
        G: i32,
        KH: i32,
        KW: i32,
    ) where
        T::I32Tensor: types::Tensor<i32, 1>,
        T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
        T::Tensor<D>: core::ops::Add<T::Tensor<D>, Output = T::Tensor<D>>,
        T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    {
        // `c_in_local`, `kh` and `kw` are in scope here, bound by the generated
        // decode. Using them for addressing is constraint C1 and is not built
        // yet, so this body only has to prove they exist and are typed.
        acc = acc + x.tensor;
    }

    /// The generated decode is the same innermost-first arithmetic conv2d writes
    /// by hand (teenygrad-y8aa, Option D).
    #[test]
    fn test_the_generated_loop_decode_matches_conv2ds_hand_written_one() {
        let src = LoopDecodeProbeForward::<f32>::new(128).source;

        // Innermost first: kw takes the modulo, then kh, and the outermost
        // takes what is left with no trailing division -- as the flat
        // program_id decode does.
        assert!(
            src.contains("let kw = __tile_loop_rem % (KW)"),
            "kw is the innermost axis: {src}"
        );
        assert!(
            src.contains("let kh = __tile_loop_rem % (KH)"),
            "kh is next: {src}"
        );
        assert!(
            src.contains("let c_in_local = __tile_loop_rem ;")
                || src.contains("let c_in_local = __tile_loop_rem;"),
            "the outermost axis takes the remainder, undivided: {src}"
        );

        // The count is the product of the extents, so no separate `count` was
        // needed -- and the parenthesised `(C_IN / G)` survives, which is the
        // reason an extent is an expression.
        assert!(
            src.contains("(C_IN / G)"),
            "an axis extent may be an expression: {src}"
        );
    }

    /// The decode probe compiles with the real teenyc, loop axes and all.
    #[test]
    fn test_the_loop_decode_probe_compiles() {
        let kernel = LoopDecodeProbeForward::<f32>::new(128);
        let target = teeny_runtime::reference_target();
        let compiled = teeny_runtime::compile_kernel(&kernel, &target, true, false)
            .expect("a generated loop with a decoded index must compile");
        assert!(!compiled.is_empty(), "compile produced no artifact");
    }

    /// Probe for a scalar loop-indexed operand -- the first half of constraint
    /// C1 (teenygrad-y8aa).
    ///
    /// Shaped like conv2d's weight, whose whole per-iteration handling is
    ///
    /// ```text
    /// let w_idx = ((c_out * c_in_per_group + c_in_local) * KH + kh) * KW + kw;
    /// let w_off = T::arange(0, 1) + w_idx;
    /// T::broadcast_to(T::load(w_ptr.add_offsets(w_off), ..), &[BLOCK_OW])
    /// ```
    ///
    /// `w` is an `In<Tile<..>>` but is deliberately NOT loaded by the prelude:
    /// its address depends on the loop index, which is exactly what the
    /// load-once-up-front prelude cannot express. The generated loop loads it
    /// per iteration instead.
    ///
    /// Note the index list mixes origins -- `kh` and `kw` come from the loop
    /// decode, and a grid-derived index would sit alongside them. Both are in
    /// scope by the time the generated load runs, which is why one list works.
    #[tiled_kernel]
    #[tile_loop(trip_count = [KH, KW], axes = [kh = KH, kw = KW], generate)]
    #[tile_carry(acc = [BLOCK_N])]
    pub fn scalar_operand_probe_forward<T: Triton, D: Num, const BLOCK_N: i32>(
        #[tile(block = BLOCK_N, extent = n_elements)] x: In<Tile<T, D>>,
        #[tile_loop_scalar(index = [kh = KH, kw = KW])] w: In<Tile<T, D>>,
        #[tile(block = BLOCK_N, extent = n_elements)] y: Out<Tile<T, D>>,
        n_elements: i32,
        KH: i32,
        KW: i32,
    ) where
        T::I32Tensor: types::Tensor<i32, 1>,
        T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
        T::Tensor<D>: core::ops::Add<T::Tensor<D>, Output = T::Tensor<D>>,
        T::Tensor<D>: core::ops::Mul<T::Tensor<D>, Output = T::Tensor<D>>,
        T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    {
        // `w` is the broadcast scalar for this iteration; `x.tensor` is loaded
        // once by the prelude, since its address does not depend on the loop.
        acc = acc + x.tensor * w;
    }

    /// The scalar operand is loaded inside the loop, at a row-major offset, and
    /// broadcast to the carry's shape (teenygrad-y8aa).
    #[test]
    fn test_a_scalar_loop_operand_is_loaded_per_iteration() {
        // The emitted source is pretty-printed and line-wrapped, so compare on
        // whitespace-collapsed text rather than literal substrings.
        let raw = ScalarOperandProbeForward::<f32>::new(128).source;
        let src: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");

        let loop_at = src
            .find("for __tile_loop_idx")
            .expect("the loop is generated");
        let load = src
            .find("arange(0, 1) + ((kw) + (kh) * (KW))")
            .unwrap_or_else(|| {
                panic!("the scalar's offset is the row-major fold of its index list: {src}")
            });
        assert!(
            loop_at < load,
            "a loop-indexed operand must be loaded INSIDE the loop, not by the \
             prelude: that is the whole of constraint C1"
        );

        // Broadcast to the carry's shape, so the product with the data tile is
        // well-typed by construction.
        assert!(
            src.contains("broadcast_to"),
            "the scalar is broadcast: {src}"
        );

        // And the prelude did not also load it up front.
        let prelude = &src[..loop_at];
        assert!(
            !prelude.contains("w.add_offsets"),
            "the prelude must skip a `#[tile_loop_scalar]` parameter: {prelude}"
        );
    }

    /// The scalar-operand probe compiles with the real teenyc.
    #[test]
    fn test_the_scalar_operand_probe_compiles() {
        let kernel = ScalarOperandProbeForward::<f32>::new(128);
        let target = teeny_runtime::reference_target();
        let compiled = teeny_runtime::compile_kernel(&kernel, &target, true, false)
            .expect("a per-iteration scalar operand must compile");
        assert!(!compiled.is_empty(), "compile produced no artifact");
    }

    /// Probe for a windowed loop-indexed operand -- the second half of
    /// constraint C1, and the whole of what teenygrad-1nr.18.2 re-scoped into
    /// teenygrad-y8aa.
    ///
    /// Shaped like conv2d's `x`: two plain axes, one *scalar* windowed axis and
    /// one *blocked* windowed axis, which is every case the conv and pool family
    /// has. conv2d writes it by hand as
    ///
    /// ```text
    /// let ih       = oh * STRIDE_H + kh - PAD_H;
    /// let iw_range = ow_range * STRIDE_W + kw - PAD_W;
    /// let ih_t     = ow_range * 0 + ih;
    /// let mask     = ow_mask & ih_t.ge(0) & ih_t.lt(H) & iw_range.ge(0) & iw_range.lt(W);
    /// T::load(x_ptr.add_offsets(iw_range + ((b * C_IN + c_in) * H * W + ih * W)), Some(mask), ..)
    /// ```
    ///
    /// Coordinates are expressions, not names: a windowed coordinate mixes a
    /// grid index, a loop index and two consts, all of which are in scope where
    /// the generated read sits. Taking the expression keeps the macro out of
    /// inferring which loop axis pairs with which window -- ambiguous the moment
    /// two axes share a kernel const, as a square kernel would.
    #[tiled_kernel]
    #[tile_loop(trip_count = [KH, KW], axes = [kh = KH, kw = KW], generate)]
    #[tile_carry(acc = [BLOCK_N])]
    pub fn windowed_operand_probe_forward<T: Triton, D: Num, const BLOCK_N: i32>(
        // BOTH declarations, which is conv2d's shape: `#[tile(..)]` says what
        // the operand *is* -- and is what teenygrad-1tl.7 put there, windows
        // included -- while `#[tile_loop_tile(..)]` says how to *read* it.
        // Different jobs, so the spec keeps the operand and only the prelude
        // skips it.
        #[tile(name = "C", extent = C)]
        #[tile(extent = H, window(stride = STRIDE_H, pad = PAD_H, kernel = KH, output = OH))]
        #[tile(
            block = BLOCK_N,
            extent = W,
            window(stride = STRIDE_W, pad = PAD_W, kernel = KW, output = OW)
        )]
        #[tile_loop_tile(
            index = [
                tile_c = C,
                (tile_oh * STRIDE_H + kh - PAD_H) = H,
                (__tile_range * STRIDE_W + kw - PAD_W) = W
            ],
            bounds = [1, 2]
        )]
        x: In<Tile<T, D>>,
        #[tile(name = "C", extent = C)]
        #[tile(name = "OH", extent = OH)]
        #[tile(block = BLOCK_N, extent = OW)]
        y: Out<Tile<T, D>>,
        C: i32,
        H: i32,
        W: i32,
        OH: i32,
        OW: i32,
        KH: i32,
        KW: i32,
        STRIDE_H: i32,
        STRIDE_W: i32,
        PAD_H: i32,
        PAD_W: i32,
    ) where
        T::I32Tensor: types::Tensor<i32, 1>,
        T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
        T::BoolTensor: core::ops::BitAnd<Output = T::BoolTensor>,
        T::Tensor<D>: core::ops::Add<T::Tensor<D>, Output = T::Tensor<D>>,
        T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    {
        // `x` is this iteration's windowed tile, already masked.
        acc = acc + x;
    }

    /// The windowed read is generated inside the loop, with the window
    /// arithmetic, the boundary mask and the row-major offset conv2d writes by
    /// hand (teenygrad-y8aa).
    #[test]
    fn test_a_windowed_loop_operand_matches_conv2ds_hand_written_read() {
        let raw = WindowedOperandProbeForward::<f32>::new(128).source;
        let src: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");

        let loop_at = src
            .find("for __tile_loop_idx")
            .expect("the loop is generated");

        // Row-major offset over [C, H, W] with the windowed coordinates
        // substituted. Innermost coordinate first, so it reads
        // `iw + (ih + c * H) * W` -- the same value as conv2d's
        // `iw_range + ((b * C_IN + c_in) * H * W + ih * W)`, and in the same
        // order, because only `Tensor + i32` has an impl.
        let off = src
            .find(
                "((__tile_range * STRIDE_W + kw - PAD_W)) + \
                 (((tile_oh * STRIDE_H + kh - PAD_H)) + (tile_c) * (H)) * (W)",
            )
            .unwrap_or_else(|| panic!("row-major offset over the windowed coords: {src}"));
        assert!(loop_at < off, "the read must be inside the loop");

        // Each checked coordinate is bound once, then tested -- conv2d binds
        // `ih` and `iw_range` for the same reason.
        //
        // The scalar one is splatted through `__tile_range * 0 + ..`, which
        // conv2d documents as load bearing rather than stylistic: a scalar
        // `if`/`continue` there trips a compiler phi-node bug. The *vector*
        // one is not splatted, because it is already a tensor -- splatting it
        // would cost a multiply and an add per check for nothing, and conv2d
        // splats only its scalar.
        assert!(
            src.contains("__tile_coord_1 = __tile_range * 0 + ((tile_oh * STRIDE_H + kh - PAD_H))"),
            "the scalar windowed coord is bound and splatted: {src}"
        );
        assert!(
            src.contains("__tile_coord_2 = (__tile_range * STRIDE_W + kw - PAD_W)"),
            "the vector windowed coord is bound and NOT splatted: {src}"
        );
        assert!(
            src.contains("__tile_coord_1.ge(0) & __tile_coord_1.lt(H)"),
            "the bound coordinate is what gets tested, not a re-spelled copy: {src}"
        );
        assert!(
            src.contains(".ge(0)") && src.contains(".lt(H)"),
            "H is bounds-checked: {src}"
        );
        assert!(src.contains(".lt(W)"), "W is bounds-checked: {src}");

        // The output tile's own mask is still folded in, as conv2d folds
        // `ow_mask`.
        assert!(
            src.contains("in_bounds &"),
            "the output tile's mask is part of the read mask: {src}"
        );

        // A plain axis indexed by a grid index gets no bounds check: it is in
        // bounds by construction, and conv2d masks only H and W.
        assert!(
            !src.contains(".lt(C)"),
            "a plain axis must not be bounds-checked: {src}"
        );

        // And the prelude did not load it up front.
        assert!(
            !src[..loop_at].contains("x.add_offsets"),
            "a loop-indexed operand must not also be loaded by the prelude"
        );
    }

    /// A loop-indexed operand stays in the spec while the prelude skips it
    /// (teenygrad-y8aa).
    ///
    /// The two declarations answer different questions, and collapsing them
    /// would silently delete the operand from its kernel's spec -- which for
    /// conv2d is exactly the windowed axes teenygrad-1tl.7 spent the effort to
    /// put there.
    #[test]
    fn test_a_loop_indexed_operand_is_still_in_the_spec() {
        let spec = WindowedOperandProbeForward::<f32>::tile_spec();
        spec.validate()
            .expect("a derived spec must be self-consistent");

        let x = spec
            .inputs
            .iter()
            .find(|i| i.param == "x")
            .expect("a loop-indexed operand must still appear among the spec's inputs");
        assert_eq!(x.rank, 3, "its declared axes are intact");

        // And its windows survived, which is the part that matters: the spec
        // says the input region is larger than the output tile.
        let windowed: Vec<(&str, &str)> = x
            .axes
            .iter()
            .filter(|a| a.window.is_some())
            .map(|a| (a.extent_param, a.block_const))
            .collect();
        assert_eq!(
            windowed,
            vec![("H", "1"), ("W", "BLOCK_N")],
            "both windowed axes survive, H with the fixed block of 1"
        );

        // Meanwhile the prelude did not load it -- the other half of the split.
        let raw = WindowedOperandProbeForward::<f32>::new(128).source;
        let src: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        let loop_at = src
            .find("for __tile_loop_idx")
            .expect("the loop is generated");
        assert!(
            !src[..loop_at].contains("x.add_offsets"),
            "the prelude must still skip it"
        );
    }

    /// The windowed-operand probe compiles with the real teenyc.
    #[test]
    fn test_the_windowed_operand_probe_compiles() {
        let kernel = WindowedOperandProbeForward::<f32>::new(128);
        let target = teeny_runtime::reference_target();
        let compiled = teeny_runtime::compile_kernel(&kernel, &target, true, false)
            .expect("a per-iteration windowed operand must compile");
        assert!(!compiled.is_empty(), "compile produced no artifact");
    }

    /// Two declared axes give a fixed-rank spec with one binding per
    /// *blocked* axis -- the untiled one is named in `untiled_dims`, the
    /// same shape the raw-pointer path produces. Contrast the single-axis
    /// kernels (`relu`/`silu`), whose spec takes a runtime `rank` because
    /// the same flat kernel really does apply at any rank.
    /// Copy over a rank-2 `[M, N]` tensor with **both** axes block-tiled, so
    /// the tile is genuinely 2-D (teenygrad-1nr.18.5).
    ///
    /// The prelude gives each blocked axis its own range, broadcast into its
    /// own dimension -- `[BLOCK_M, 1]` and `[1, BLOCK_N]` -- then combines them
    /// by stride and conjoins their bounds. No extra where-clause is needed for
    /// that: `Triton` already bounds `BoolTensor: BitAnd` at the trait level,
    /// so conv2d_forward's own declaration of it is redundant. The rank bound
    /// stays
    /// `Tensor<i32, 1>`: every tensor type in the tree implements
    /// `Tensor<D, RANK>` generically over RANK, so it holds at rank 2 too.
    #[tiled_kernel]
    pub fn two_blocked_axes_probe_forward<
        T: Triton,
        D: Num,
        const BLOCK_M: i32,
        const BLOCK_N: i32,
    >(
        #[tile(block = BLOCK_M, extent = M)]
        #[tile(block = BLOCK_N, extent = N)]
        x: In<Tile<T, D>>,
        #[tile(block = BLOCK_M, extent = M)]
        #[tile(block = BLOCK_N, extent = N)]
        y: Out<Tile<T, D>>,
        M: i32,
        N: i32,
    ) where
        T::I32Tensor: types::Tensor<i32, 1>,
        T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
        T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    {
        T::store(y.tensor, x.tensor, x.mask, &[], None, None);
    }

    /// Both axes are blocked, so both become `TileAxisBinding`s -- the shape a
    /// GEMM needs, and what `teenygrad-1tl.10` is waiting on.
    #[test]
    fn test_two_blocked_axes_give_a_binding_each() {
        let spec = TwoBlockedAxesProbeForward::<f32>::tile_spec();

        for tensor in [spec.inputs[0], spec.outputs[0]] {
            assert_eq!(tensor.rank, 2);
            assert_eq!(
                tensor.axes.len(),
                2,
                "both axes carry `block = ..`, so neither is untiled"
            );
            assert_eq!(tensor.axes[0].block_const, "BLOCK_M");
            assert_eq!(tensor.axes[0].extent_param, "M");
            assert_eq!(tensor.axes[0].dims, &[0]);
            assert_eq!(tensor.axes[1].block_const, "BLOCK_N");
            assert_eq!(tensor.axes[1].extent_param, "N");
            assert_eq!(tensor.axes[1].dims, &[1]);
            assert!(
                tensor.untiled_dims.is_empty(),
                "nothing is left untiled by a fully blocked tile"
            );
        }

        spec.validate()
            .expect("a derived spec must be self-consistent");
    }

    #[test]
    fn test_two_declared_axes_give_a_fixed_rank_spec() {
        let spec = MultiAxisProbeForward::<f32>::tile_spec();

        assert_eq!(spec.inputs.len(), 1);
        assert_eq!(spec.outputs.len(), 1);
        assert_eq!(spec.loop_spec, None);

        let x = spec.inputs[0];
        let y = spec.outputs[0];
        assert_eq!((x.param, y.param), ("x", "y"));

        for tensor in [x, y] {
            assert_eq!(tensor.rank, 2, "declared by the signature, not per node");
            assert_eq!(tensor.untiled_dims, &["H"]);
            assert_eq!(tensor.axes.len(), 1, "only W carries `block = ..`");
            assert_eq!(tensor.axes[0].dims, &[1], "W is the innermost dim");
            assert_eq!(tensor.axes[0].block_const, "BLOCK_W");
            assert_eq!(tensor.axes[0].extent_param, "W");
            assert_eq!(tensor.axes[0].window, None);
        }

        spec.validate()
            .expect("a derived spec must be self-consistent");
    }
}
