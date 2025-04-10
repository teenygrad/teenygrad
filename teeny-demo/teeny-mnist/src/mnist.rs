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

pub async fn fetch_mnist_train_data(cache_dir: &str) -> Result<(String, String)> {
    let images_path = format!("{}/train-images-idx3-ubyte.gz", cache_dir);
    let labels_path = format!("{}/train-labels-idx1-ubyte.gz", cache_dir);

    check_cache_dir(cache_dir).await?;

    download_file(MNIST_TRAIN_IMAGES, &images_path).await?;
    download_file(MNIST_TRAIN_LABELS, &labels_path).await?;

    Ok((images_path, labels_path))
}

pub async fn fetch_mnist_test_data(cache_dir: &str) -> Result<(String, String)> {
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
