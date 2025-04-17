/*
 * Copyright (C) 2025 SpinorML Ltd.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use anyhow::Result;
use flate2::read::GzDecoder;
use smol::fs::File;
use smol::io::AsyncWriteExt;
use std::io::Read;
use teeny_tensor::tensor::Tensor;
use teeny_tensor::tensor::memory::MemoryTensor;

const MNIST_TRAIN_IMAGES: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/t10k-images-idx3-ubyte.gz?raw=true";
const MNIST_TRAIN_LABELS: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/t10k-labels-idx1-ubyte.gz?raw=true";

const MNIST_TEST_IMAGES: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/train-images-idx3-ubyte.gz?raw=true";
const MNIST_TEST_LABELS: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/train-labels-idx1-ubyte.gz?raw=true";

pub async fn read_mnist_train_data(
    cache_dir: &str,
) -> Result<(impl Tensor<f32>, impl Tensor<f32>)> {
    let (images_path, labels_path) = fetch_mnist_train_data(cache_dir).await?;

    let images = read_mnist_images(images_path).await?;
    let labels = read_mnist_labels(labels_path).await?;

    Ok((images, labels))
}

pub async fn read_mnist_test_data(cache_dir: &str) -> Result<(impl Tensor<f32>, impl Tensor<f32>)> {
    let (images_path, labels_path) = fetch_mnist_test_data(cache_dir).await?;

    let images = read_mnist_images(images_path).await?;
    let labels = read_mnist_labels(labels_path).await?;

    Ok((images, labels))
}

async fn fetch_mnist_train_data(cache_dir: &str) -> Result<(String, String)> {
    let images_path = format!("{}/train-images-idx3-ubyte.gz", cache_dir);
    let labels_path = format!("{}/train-labels-idx1-ubyte.gz", cache_dir);

    check_cache_dir(cache_dir).await?;

    download_file(MNIST_TRAIN_IMAGES, &images_path).await?;
    download_file(MNIST_TRAIN_LABELS, &labels_path).await?;

    Ok((images_path, labels_path))
}

async fn fetch_mnist_test_data(cache_dir: &str) -> Result<(String, String)> {
    let images_path = format!("{}/t10k-images-idx3-ubyte.gz", cache_dir);
    let labels_path = format!("{}/t10k-labels-idx1-ubyte.gz", cache_dir);

    check_cache_dir(cache_dir).await?;
    download_file(MNIST_TEST_IMAGES, &images_path).await?;
    download_file(MNIST_TEST_LABELS, &labels_path).await?;

    Ok((images_path, labels_path))
}

async fn check_cache_dir(cache_dir: &str) -> Result<()> {
    if !std::path::Path::new(cache_dir).exists() {
        return Err(anyhow::anyhow!(
            "Cache directory does not exist: {}",
            cache_dir
        ));
    }
    Ok(())
}

async fn download_file(url: &str, cache_path: &str) -> Result<()> {
    if std::path::Path::new(cache_path).exists() {
        return Ok(());
    }

    let client = surf::client().with(surf::middleware::Redirect::new(20));
    let mut response = client
        .get(url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to send request: {}", e))?;

    let mut file = File::create(cache_path).await?;
    let content = response
        .body_bytes()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read response body: {}", e))?;
    file.write_all(&content).await?;
    file.close().await?;

    Ok(())
}

async fn read_mnist_images(path: String) -> Result<impl Tensor<f32>> {
    let file = std::fs::File::open(path)?;
    let mut reader = GzDecoder::new(file);
    let mut images = Vec::new();
    reader.read_to_end(&mut images)?;

    let scale_factor = 255f32;

    let data = images
        .iter()
        .map(|b| *b as f32 / scale_factor)
        .collect::<Vec<f32>>();

    Ok(MemoryTensor::with_data(&[-1, 28], data))
}

async fn read_mnist_labels(path: String) -> Result<impl Tensor<f32>> {
    let file = std::fs::File::open(path)?;
    let mut reader = GzDecoder::new(file);
    let mut labels = Vec::new();
    reader.read_to_end(&mut labels)?;

    let data = labels.iter().map(|b| *b as f32).collect::<Vec<f32>>();

    Ok(MemoryTensor::with_data(&[-1, 1], data))
}
