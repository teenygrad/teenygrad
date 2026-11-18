//! Probe: does a byte-wide comparison kernel compile now?
//!
//! `ElemwiseEqualForward::<u8>` used to fail in `teenyc` with
//! "'tt.splat' op requires the same element type for all operands and results",
//! which is why `nn/tensor/logical.rs` is written `u8`-concrete rather than generic
//! over `D`. `ElemwiseMulForward::<u8>` compiled, so the fault was in the comparison
//! and not in `u8` itself. teenygrad/teeny#32 ("use unsigned integer ops for unsigned
//! element types") and #33 ("extend or truncate scalar int-to-int casts") both look
//! like candidates. Neither fixes it: on teenyc `18d0e1214` the control still compiles
//! and the comparison still does not.
//!
//! What the error now says is more specific than "u8 is unsupported". The kernel body
//! ends in
//!
//! ```text
//! let one = T::cast::<i32, D>(T::full::<i32>(&[BLOCK_SIZE], 1), None, false);
//! ```
//!
//! and at `D = u8` the `full` and the `cast` collapse into a single `tt.splat` that
//! takes an `i32` operand and claims an `i8` result:
//!
//! ```text
//! error: 'tt.splat' op requires the same element type for all operands and results
//! note: see current operation: %17 = "tt.splat"(%1) : (i32) -> tensor<1024xi8>
//! ```
//!
//! So the splat's result element type comes from the cast's target while its operand
//! keeps the source width, and nothing narrows the scalar in between. That is why
//! `nn/tensor/logical.rs` builds its constants with `T::full::<u8>` directly: a splat
//! whose operand and result agree needs no cast at all.
//!
//! Not reported upstream yet.

use teeny_kernels::nn::tensor::elemwise_binary::{ElemwiseEqualForward, ElemwiseMulForward};

const BLOCK_SIZE: i32 = 1024;

#[test]
#[ignore = "teenyc splats an i32 scalar into an i8 tensor; see the module note"]
fn a_byte_wide_comparison_compiles() -> anyhow::Result<()> {
    let target = teeny_runtime::reference_target();

    // The control: byte-wide arithmetic has always compiled.
    let mul = ElemwiseMulForward::<u8>::new(BLOCK_SIZE);
    let control = teeny_runtime::compile_kernel(&mul, &target, true, false);
    println!(
        "ElemwiseMulForward::<u8>   : {}",
        if control.is_ok() { "COMPILES" } else { "FAILS" }
    );
    if let Err(e) = &control {
        println!("  {e}");
    }

    let eq = ElemwiseEqualForward::<u8>::new(BLOCK_SIZE);
    let probe = teeny_runtime::compile_kernel(&eq, &target, true, false);
    match &probe {
        Ok(p) => println!("ElemwiseEqualForward::<u8> : COMPILES ({p})"),
        Err(e) => println!("ElemwiseEqualForward::<u8> : FAILS\n  {e}"),
    }

    assert!(
        control.is_ok(),
        "the control kernel stopped compiling, so this probe says nothing"
    );
    assert!(
        probe.is_ok(),
        "a byte-wide comparison still does not compile"
    );
    Ok(())
}
