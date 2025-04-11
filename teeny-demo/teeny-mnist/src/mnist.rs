/*
 * Copyright (c) SpinorML 2025. All rights reserved.
 *
 * This software and associated documentation files (the "Software") are proprietary
 * and confidential. The Software is protected by copyright laws and international
 * copyright treaties, as well as other intellectual property laws and treaties.
 *
 * No part of this Software may be reproduced, distributed, or transmitted in any
 * form or by any means, including photocopying, recording, or other electronic or
 * mechanical methods, without the prior written permission of SpinorML.
 *
 * Unauthorized copying, modification, distribution, or use of this Software is
 * strictly prohibited and may result in severe civil and criminal penalties.
 */

use anyhow::Result;
use flate2::read::GzDecoder;
use std::io::Read;
use teeny_core::tensor::memory::MemoryTensor;
use teeny_core::tensor::{ElementType, Tensor};
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

const MNIST_TRAIN_IMAGES: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/t10k-images-idx3-ubyte.gz";
const MNIST_TRAIN_LABELS: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/t10k-labels-idx1-ubyte.gz";

const MNIST_TEST_IMAGES: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/train-images-idx3-ubyte.gz";
const MNIST_TEST_LABELS: &str =
    "https://github.com/spinorml/data/blob/main/models/mnist/train-labels-idx1-ubyte.gz";

pub async fn read_mnist_train_data(
    cache_dir: &str,
) -> Result<(impl Tensor<half::f16>, impl Tensor<half::f16>)> {
    let (images_path, labels_path) = fetch_mnist_train_data(cache_dir).await?;

    let images = read_mnist_images(images_path).await?;
    let labels = read_mnist_labels(labels_path).await?;

    Ok((images, labels))
}

pub async fn read_mnist_test_data(
    cache_dir: &str,
) -> Result<(impl Tensor<half::f16>, impl Tensor<half::f16>)> {
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

    let response = reqwest::get(url).await?;
    let mut file = File::create(cache_path).await?;
    let content = response.bytes().await?;
    file.write_all(&content).await?;

    Ok(())
}

async fn read_mnist_images(path: String) -> Result<impl Tensor<half::f16>> {
    let file = std::fs::File::open(path)?;
    let mut reader = GzDecoder::new(file);
    let mut images = Vec::new();
    reader.read_to_end(&mut images)?;

    let scale_factor = half::f16::from_bits(255);

    let data = images
        .iter()
        .map(|b| half::f16::from_bits(*b as u16) / scale_factor)
        .collect::<Vec<half::f16>>();

    Ok(MemoryTensor::new(
        ElementType::FP16,
        Vec::from([data.len() as i64]),
        data,
    ))
}

async fn read_mnist_labels(path: String) -> Result<impl Tensor<half::f16>> {
    let file = std::fs::File::open(path)?;
    let mut reader = GzDecoder::new(file);
    let mut labels = Vec::new();
    reader.read_to_end(&mut labels)?;

    let data = labels
        .iter()
        .map(|b| half::f16::from_bits(*b as u16))
        .collect::<Vec<half::f16>>();

    Ok(MemoryTensor::new(
        ElementType::FP16,
        Vec::from([data.len() as i64]),
        data,
    ))
}
