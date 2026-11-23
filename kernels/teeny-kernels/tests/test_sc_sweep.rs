#![cfg(feature = "cuda")]
use teeny_core::device::Device as _;
use teeny_core::device::buffer::Buffer;
use teeny_core::model::RuntimeOp;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::errors::Result;
use teeny_kernels::nn::tensor::strided_copy::{StridedCopyForward, StridedCopyRuntimeOp};
use teeny_test::cuda as testing;

/// Host transpose of a 2-D tensor.
fn ref_transpose(x: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = x[i * c + j];
        }
    }
    out
}

#[test]
fn test_strided_copy_transpose_sweep() -> Result<()> {
    dotenv::dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let device = env.device;
    let kernel = StridedCopyForward::<f32>::new(1024);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<StridedCopyForward<f32>>(&ptx)?;

    println!("rows(=inner out extent)  wrong");
    for rows in 2..=13usize {
        let cols = 5usize;
        // transpose([rows, cols]) -> out [cols, rows], innermost out extent = rows
        let x: Vec<f32> = (0..rows * cols).map(|i| i as f32 + 0.5).collect();
        let want = ref_transpose(&x, rows, cols);
        let op = StridedCopyRuntimeOp::<f32>::transpose(1024, &[rows, cols], &[1, 0]);
        let out_shape = [cols, rows];
        let n = rows * cols;
        let a = op.kernel_scalars(n as i32);

        let mut x_buf = device.buffer::<f32>(n)?;
        let out_buf = device.buffer::<f32>(n)?;
        x_buf.to_device(&x)?;
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
            ),
        )?;
        let mut got = vec![0.0f32; n];
        out_buf.to_host(&mut got)?;
        let wrong = (0..n).filter(|&i| got[i] != want[i]).count();
        println!("{rows:>23}  {wrong:>5}   rows%4={}", rows % 4);
    }
    Ok(())
}
