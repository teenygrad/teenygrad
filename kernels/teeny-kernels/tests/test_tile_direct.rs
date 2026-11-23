// Launches tile_forward directly, bypassing lowering and the model executor.
//
// test_tile.rs runs Tile through RuntimeModel, where it is correct on RISC-V but
// diverges on CUDA for specific lanes. This isolates which layer is responsible: the
// same kernel, the same scalars from the same RuntimeOp, launched by hand.

#![cfg(feature = "cuda")]

use teeny_core::device::Device as _;
use teeny_core::device::buffer::Buffer;
use teeny_core::model::RuntimeOp;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_kernels::nn::tensor::tile::{TileForward, TileRuntimeOp};
use teeny_test::cuda as testing;

const BLOCK: i32 = 1024;

/// ONNX Tile on the host.
fn reference(x: &[f32], in_shape: &[usize], repeats: &[i64]) -> (Vec<f32>, Vec<usize>) {
    let rank = in_shape.len();
    let out_shape: Vec<usize> = in_shape
        .iter()
        .zip(repeats)
        .map(|(&d, &r)| d * r as usize)
        .collect();
    let mut is = vec![1usize; rank];
    let mut os = vec![1usize; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        is[d] = is[d + 1] * in_shape[d + 1];
        os[d] = os[d + 1] * out_shape[d + 1];
    }
    let n: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; n];
    for (flat, slot) in out.iter_mut().enumerate() {
        let mut src = 0usize;
        let mut rem = flat;
        for d in 0..rank {
            let c = rem / os[d];
            rem -= c * os[d];
            src += (c % in_shape[d]) * is[d];
        }
        *slot = x[src];
    }
    (out, out_shape)
}

fn check(in_shape: &[usize], repeats: &[i64]) -> Result<usize> {
    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;

    let n_in: usize = in_shape.iter().product();
    let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();
    let (want, out_shape) = reference(&x, in_shape, repeats);

    let op = TileRuntimeOp::<f32>::new(BLOCK, in_shape, repeats).expect("valid tile");
    let n: usize = out_shape.iter().product();
    let a = op.kernel_scalars(n as i32);
    assert_eq!(a.len(), 18, "kernel ABI: {} scalars", a.len());

    let mut x_buf = device.buffer::<f32>(x.len())?;
    let out_buf = device.buffer::<f32>(n)?;
    x_buf.to_device(&x)?;

    let kernel = TileForward::<f32>::new(BLOCK);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<TileForward<f32>>(&ptx)?;

    let grid = op.grid(&out_shape);
    let cfg = testing::launch_config_with_grid(grid[0] as usize, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            a[10],
            a[11],
            a[12],
            a[13],
            a[14],
            a[15],
            a[16],
            a[17],
        ),
    )?;

    let mut got = vec![0.0f32; n];
    out_buf.to_host(&mut got)?;

    let bad: Vec<usize> = (0..n).filter(|&i| got[i] != want[i]).collect();
    if !bad.is_empty() {
        println!(
            "tile {in_shape:?} x {repeats:?}: {} of {n} wrong at {bad:?}",
            bad.len()
        );
        println!(
            "  device idx = {:?}",
            got.iter().map(|v| v - 0.5).collect::<Vec<_>>()
        );
        println!(
            "  want   idx = {:?}",
            want.iter().map(|v| v - 0.5).collect::<Vec<_>>()
        );
    } else {
        println!("tile {in_shape:?} x {repeats:?}: {n} elements exact");
    }
    Ok(bad.len())
}

/// Sweeps the innermost output extent, holding everything else fixed, to test whether
/// divisibility by 4 is what separates the passing cases from the failing ones.
#[test]
fn test_sweep_innermost_extent() -> Result<()> {
    println!("inner_extent  n_elements  wrong");
    for inner in 2..=13usize {
        // in = [2, inner], repeats = [2, 1] -> out = [4, inner].
        let in_shape = [2usize, inner];
        let repeats = [2i64, 1];
        let n = 4 * inner;
        let wrong = check_quiet(&in_shape, &repeats)?;
        println!("{inner:>12}  {n:>10}  {wrong:>5}   inner%4={}", inner % 4);
    }
    Ok(())
}

fn check_quiet(in_shape: &[usize], repeats: &[i64]) -> Result<usize> {
    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let n_in: usize = in_shape.iter().product();
    let x: Vec<f32> = (0..n_in).map(|i| i as f32 + 0.5).collect();
    let (want, out_shape) = reference(&x, in_shape, repeats);
    let op = TileRuntimeOp::<f32>::new(BLOCK, in_shape, repeats).expect("valid tile");
    let n: usize = out_shape.iter().product();
    let a = op.kernel_scalars(n as i32);
    let mut x_buf = device.buffer::<f32>(x.len())?;
    let out_buf = device.buffer::<f32>(n)?;
    x_buf.to_device(&x)?;
    let kernel = TileForward::<f32>::new(BLOCK);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<TileForward<f32>>(&ptx)?;
    let grid = op.grid(&out_shape);
    let cfg = testing::launch_config_with_grid(grid[0] as usize, &program);
    device.launch(
        &program,
        &cfg,
        (
            x_buf.as_device_ptr(),
            out_buf.as_device_ptr(),
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            a[5],
            a[6],
            a[7],
            a[8],
            a[9],
            a[10],
            a[11],
            a[12],
            a[13],
            a[14],
            a[15],
            a[16],
            a[17],
        ),
    )?;
    let mut got = vec![0.0f32; n];
    out_buf.to_host(&mut got)?;
    Ok((0..n).filter(|&i| got[i] != want[i]).count())
}

#[test]
fn test_direct_launch_matches_the_reference() -> Result<()> {
    let mut total = 0;
    total += check(&[3], &[4])?;
    total += check(&[2, 3], &[2, 2])?;
    total += check(&[2, 3], &[1, 3])?;
    total += check(&[2, 3, 2], &[2, 1, 3])?;
    total += check(&[1, 4], &[5, 1])?;
    assert_eq!(total, 0, "{total} elements wrong across all cases");
    Ok(())
}
