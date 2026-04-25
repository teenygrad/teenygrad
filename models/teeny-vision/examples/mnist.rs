/*
 * Copyright (c) 2026 Teenygrad.
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

use std::env;

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::graph::compiler::GraphCompiler;
use teeny_core::graph::{DtypeRepr, SymTensor};
use teeny_cuda::compiler::graph::CudaGraphCompiler;
use teeny_cuda::compiler::target::Capability;
use teeny_cuda::model::CudaModel;
use teeny_kernels::graph::TritonLowering;
use teeny_vision::mnist;
use tokio::io::AsyncWriteExt;

#[tokio::main]
async fn main() -> Result<()> {
    let cache_dir =
        env::var("TEENY_DATA_DIR").unwrap_or_else(|_| "/tmp/teenygrad_cache".to_string());
    let cache_dir = PathBuf::from(cache_dir);
    download_data(&cache_dir).await?;

    let (input, graph) =
        SymTensor::input(DtypeRepr::F32, vec![Some(1), Some(1), Some(28), Some(28)]);
    let _model = mnist::mnist::<f32>()(input);
    let lowering = TritonLowering::new();
    let graph_compiler = CudaGraphCompiler::new();
    let target = Target::new(Capability::Sm90);
    // let _model =
    //     graph_compiler.compile::<_, _, CudaModel<'static>>(&graph.borrow(), &lowering, &target)?;

    println!("Hello, world!: {:?}", graph.borrow().nodes);

    Ok(())
}

async fn download_data(cache_dir: &Path) -> Result<()> {
    const TEST_DATA: &str = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/test-00000-of-00001.parquet?download=true";
    const TRAIN_DATA: &str = "https://huggingface.co/datasets/ylecun/mnist/resolve/main/mnist/train-00000-of-00001.parquet?download=true";

    let test_file = cache_dir.join("mnist/mnist-test.parquet");
    let train_file = cache_dir.join("mnist/mnist-train.parquet");

    download_if_not_exists(TEST_DATA, &test_file).await?;
    download_if_not_exists(TRAIN_DATA, &train_file).await?;

    Ok(())
}

// Helper function to fetch a file and save to a local path
async fn download_if_not_exists(url: &str, path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }
    let mut response = reqwest::get(url).await?;
    response.error_for_status_ref()?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    let total_size = response.content_length().unwrap_or(0);
    let progress = if total_size > 0 {
        let bar = ProgressBar::new(total_size);
        bar.set_style(
            ProgressStyle::with_template(
                "{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, ETA {eta})",
            )?
            .progress_chars("##-"),
        );
        bar
    } else {
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(ProgressStyle::with_template(
            "{msg} {spinner} {bytes} downloaded",
        )?);
        spinner.enable_steady_tick(std::time::Duration::from_millis(120));
        spinner
    };
    progress.set_message(format!("Downloading {file_name}"));

    let mut file = tokio::fs::File::create(path).await?;
    while let Some(chunk) = response.chunk().await? {
        file.write_all(&chunk).await?;
        progress.inc(chunk.len() as u64);
    }
    file.flush().await?;
    progress.finish_with_message(format!("Downloaded {file_name}"));

    Ok(())
}
