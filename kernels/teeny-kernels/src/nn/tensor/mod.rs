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
