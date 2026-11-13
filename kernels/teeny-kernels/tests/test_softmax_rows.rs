//! Softmax rows whose width is not a power of two, and shapes above rank 2.
//!
//! The kernel normalizes a flattened `[rows, cols]` view, one CTA per row, and two
//! things used to go wrong away from the case the fixtures cover:
//!
//! - `BLOCK_SIZE` is `n_cols` rounded up to a power of two, and the load and the store
//!   were unmasked, so the lanes past a row joined its sum and were written over the
//!   next row's start. On the last row both ran past the end of the buffer.
//! - `grid` was `[output_shape[0], 1, 1]` and `pack_args` read `n_cols` as
//!   `output_shape[1]`, which is the row count and width only at rank 2.
//!
//! The launches here pass the extents directly, so they pin the kernel itself; the
//! shape reading is pinned by `rows_and_cols`'s own tests.
#![cfg(feature = "hardware")]

use dotenv::dotenv;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_kernels::nn::activation::softmax::SoftmaxForward;

const PTX_LAUNCH_THREADS_X: u32 = 128;

/// Marks memory the kernel must not write.
const SENTINEL: f32 = -7.0;

/// Reference softmax of each row, computed the stable way.
fn reference(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max = row.iter().fold(f32::NEG_INFINITY, |m, v| m.max(*v));
        let exps: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (c, e) in exps.iter().enumerate() {
            out[r * cols + c] = e / sum;
        }
    }
    out
}

fn run(rows: usize, cols: usize) -> anyhow::Result<()> {
    dotenv().ok();
    let device = teeny_runtime::open()?;

    let n = rows * cols;
    // Distinct values with a spread, so a row that borrowed a neighbour's elements
    // cannot land on the right answer by symmetry.
    let x: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.25 - (n as f32) / 8.0)
        .collect();
    let want = reference(&x, rows, cols);

    // The output buffer carries one extra row, filled with a sentinel beforehand, so a
    // store that runs past the last row shows up as a sentinel that changed.
    let mut x_buf = device.buffer::<f32>(n)?;
    let mut y_buf = device.buffer::<f32>(n + cols)?;
    let mut y_host = vec![0.0f32; n + cols];
    x_buf.to_device(&x)?;
    y_buf.to_device(&vec![SENTINEL; n + cols])?;

    let block_size = cols.next_power_of_two() as i32;
    let kernel = SoftmaxForward::<f32>::new(block_size);
    let target = teeny_runtime::default_target(&device)?;
    let ptx_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<SoftmaxForward<f32>>(&ptx_path)?;

    let cfg = teeny_runtime::launch_config_custom(
        [rows as u32, 1, 1],
        [PTX_LAUNCH_THREADS_X, 1, 1],
        [1, 1, 1],
    );
    let args = (
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        rows as i32,
        cols as i32,
    );
    device.launch(&program, &cfg, args)?;
    y_buf.to_host(&mut y_host)?;

    for i in 0..n {
        assert!(
            (y_host[i] - want[i]).abs() < 1e-5,
            "[{rows}, {cols}] element {i} (row {}, col {}): got {}, want {}",
            i / cols,
            i % cols,
            y_host[i],
            want[i]
        );
    }
    // The spare row must be untouched: the kernel has no business past the last row.
    for (i, v) in y_host[n..].iter().enumerate() {
        assert_eq!(
            *v, SENTINEL,
            "[{rows}, {cols}] wrote {v} at {i} of the row past the output"
        );
    }
    // Every row must sum to one, which a borrowed element would break even where the
    // element-wise comparison is loose.
    for r in 0..rows {
        let sum: f32 = y_host[r * cols..(r + 1) * cols].iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "[{rows}, {cols}] row {r} sums to {sum}"
        );
    }
    Ok(())
}

#[test]
fn a_row_width_that_is_a_power_of_two() -> anyhow::Result<()> {
    run(4, 8)
}

/// `BLOCK_SIZE` is 8 for 6 columns, so two lanes per row fall outside it.
#[test]
fn a_row_width_that_is_not_a_power_of_two() -> anyhow::Result<()> {
    run(4, 6)
}

/// One row, so the overrun has nothing after it but the end of the buffer.
#[test]
fn a_single_row_of_an_odd_width() -> anyhow::Result<()> {
    run(1, 5)
}

/// Wider than one block of threads, with a ragged tail.
#[test]
fn many_rows_of_a_ragged_width() -> anyhow::Result<()> {
    run(33, 130)
}
