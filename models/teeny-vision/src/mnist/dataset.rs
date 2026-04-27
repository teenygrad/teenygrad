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

//! Reader for HuggingFace MNIST parquet files.
//!
//! Schema (from `ylecun/mnist`):
//!   - `image`: struct<bytes: binary, path: utf8>  — PNG-encoded 28×28 grayscale
//!   - `label`: int64                              — class 0-9

use std::{fs::File, path::{Path, PathBuf}};

use anyhow::{anyhow, Result};
use arrow_array::{Array, BinaryArray, Int64Array, LargeBinaryArray, StructArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// A loaded batch of MNIST samples.
pub struct MnistBatch {
    /// Pixel data in NCHW layout `[batch, 1, 28, 28]`, normalised to `[0.0, 1.0]`.
    pub images: Vec<f32>,
    /// Integer class labels `[0, 9]`, one per sample.
    pub labels: Vec<u8>,
    /// Number of samples in this batch.
    pub batch_size: usize,
}

impl MnistBatch {
    /// Total number of f32 elements: `batch_size × 1 × 28 × 28`.
    pub fn image_numel(&self) -> usize {
        self.batch_size * 28 * 28
    }
}

/// Reads MNIST samples from a HuggingFace parquet file.
pub struct MnistDataset {
    path: PathBuf,
    num_rows: usize,
}

impl MnistDataset {
    /// Open a parquet file and read its metadata.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let num_rows = builder.metadata().file_metadata().num_rows() as usize;
        Ok(Self { path, num_rows })
    }

    pub fn len(&self) -> usize {
        self.num_rows
    }

    pub fn is_empty(&self) -> bool {
        self.num_rows == 0
    }

    /// Read `count` consecutive rows starting at `offset` (0-indexed).
    ///
    /// The returned batch is truncated if `offset + count > len()`.
    pub fn read_batch(&self, offset: usize, count: usize) -> Result<MnistBatch> {
        let actual = count.min(self.num_rows.saturating_sub(offset));
        if actual == 0 {
            return Err(anyhow!(
                "offset {offset} is at or past the end of the dataset (len={})",
                self.num_rows
            ));
        }

        let file = File::open(&self.path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_offset(offset)
            .with_limit(actual)
            .with_batch_size(actual)
            .build()?;

        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("parquet reader returned no record batches"))??;

        decode_batch(&batch, actual)
    }
}

fn decode_batch(batch: &arrow_array::RecordBatch, n: usize) -> Result<MnistBatch> {
    // ── image column — struct<bytes: binary|large_binary, path: utf8> ──────────
    let image_struct = batch
        .column_by_name("image")
        .ok_or_else(|| anyhow!("column 'image' not found in parquet file"))?
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| anyhow!("column 'image' is not a StructArray"))?;

    // The `bytes` child may be Binary or LargeBinary depending on how HF serialised it.
    let bytes_field = image_struct
        .column_by_name("bytes")
        .ok_or_else(|| anyhow!("image struct has no 'bytes' field"))?;

    // ── label column — int64 ───────────────────────────────────────────────────
    let label_col = batch
        .column_by_name("label")
        .ok_or_else(|| anyhow!("column 'label' not found in parquet file"))?
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| anyhow!("column 'label' is not Int64Array"))?;

    let mut images = Vec::with_capacity(n * 28 * 28);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let png_bytes: &[u8] = if let Some(col) = bytes_field.as_any().downcast_ref::<BinaryArray>() {
            col.value(i)
        } else if let Some(col) = bytes_field.as_any().downcast_ref::<LargeBinaryArray>() {
            col.value(i)
        } else {
            return Err(anyhow!("'bytes' field has unexpected Arrow type: {:?}", bytes_field.data_type()));
        };

        let img = image::load_from_memory(png_bytes)
            .map_err(|e| anyhow!("failed to decode PNG for sample {i}: {e}"))?
            .into_luma8();

        if img.width() != 28 || img.height() != 28 {
            return Err(anyhow!(
                "unexpected image dimensions for sample {i}: {}×{}",
                img.width(), img.height()
            ));
        }

        for px in img.pixels() {
            images.push(px.0[0] as f32 / 255.0);
        }

        labels.push(label_col.value(i) as u8);
    }

    Ok(MnistBatch { images, labels, batch_size: n })
}
