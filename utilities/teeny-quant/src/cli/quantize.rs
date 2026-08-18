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

//! `teeny-quant quantize`: reads a `.safetensors` checkpoint and writes a quantized one.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;

use crate::format::{self, PackedTensorInfo};
use crate::quant::{Fp8Variant, Granularity, Scheme};
use crate::read::read_f32;
use crate::write::{OutputTensor, write_safetensors};

/// `teeny-quant quantize` arguments.
#[derive(Args, Debug)]
pub struct QuantizeArgs {
    /// Input `.safetensors` checkpoint.
    #[arg(long)]
    pub input: PathBuf,

    /// Output `.safetensors` path.
    #[arg(long)]
    pub output: PathBuf,

    /// Quantization scheme.
    #[arg(long, value_enum)]
    pub scheme: SchemeArg,

    /// Use asymmetric (rather than symmetric) affine quantization. Ignored for `--scheme fp8`,
    /// which is always amax-scaled/symmetric.
    #[arg(long)]
    pub asymmetric: bool,

    /// Quantization granularity.
    #[arg(long, value_enum, default_value = "channel")]
    pub granularity: GranularityArg,

    /// Axis for `--granularity channel`/`group` (default `0`, the output-channel axis for
    /// `Linear`/`Conv` weights shaped `[out, in, ...]`). Group-wise quantization typically wants
    /// the reduction axis instead, e.g. `--axis 1`.
    #[arg(long, default_value_t = 0)]
    pub axis: usize,

    /// Elements per group; required for `--granularity group`.
    #[arg(long)]
    pub group_size: Option<usize>,

    /// FP8 encoding; only used for `--scheme fp8`.
    #[arg(long, value_enum, default_value = "e4m3")]
    pub fp8_variant: Fp8VariantArg,
}

/// `--scheme` values.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum SchemeArg {
    /// `Scheme::Int8`.
    #[value(name = "int8")]
    Int8,
    /// `Scheme::Int4`.
    #[value(name = "int4")]
    Int4,
    /// `Scheme::Fp8`.
    #[value(name = "fp8")]
    Fp8,
}

/// `--granularity` values.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum GranularityArg {
    /// `Granularity::PerTensor`.
    #[value(name = "tensor")]
    Tensor,
    /// `Granularity::PerChannel`.
    #[value(name = "channel")]
    Channel,
    /// `Granularity::Group`.
    #[value(name = "group")]
    Group,
}

/// `--fp8-variant` values.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum Fp8VariantArg {
    /// `Fp8Variant::E4M3`.
    #[value(name = "e4m3")]
    E4M3,
    /// `Fp8Variant::E5M2`.
    #[value(name = "e5m2")]
    E5M2,
}

/// Runs `teeny-quant quantize`.
pub fn run(args: QuantizeArgs) -> Result<()> {
    let scheme = match args.scheme {
        SchemeArg::Int8 => Scheme::Int8 {
            symmetric: !args.asymmetric,
        },
        SchemeArg::Int4 => Scheme::Int4 {
            symmetric: !args.asymmetric,
        },
        SchemeArg::Fp8 => Scheme::Fp8 {
            variant: match args.fp8_variant {
                Fp8VariantArg::E4M3 => Fp8Variant::E4M3,
                Fp8VariantArg::E5M2 => Fp8Variant::E5M2,
            },
        },
    };

    let granularity = match args.granularity {
        GranularityArg::Tensor => Granularity::PerTensor,
        GranularityArg::Channel => Granularity::PerChannel { axis: args.axis },
        GranularityArg::Group => {
            let group_size = args
                .group_size
                .context("--group-size is required for --granularity group")?;
            Granularity::Group {
                axis: args.axis,
                group_size,
            }
        }
    };

    let mapped = teeny_data::safetensors::SafeTensors::from_pretrained(&args.input)
        .with_context(|| format!("failed to open '{}'", args.input.display()))?;
    let tensors = mapped.tensors().with_context(|| {
        format!(
            "failed to read tensor headers from '{}'",
            args.input.display()
        )
    })?;

    let mut outputs: HashMap<String, OutputTensor> = HashMap::new();
    let mut ignored = Vec::new();
    let mut packed_int4 = HashMap::new();
    let mut quantized_count = 0usize;

    for name in tensors.names() {
        let view = tensors
            .tensor(name)
            .with_context(|| format!("reading tensor '{name}'"))?;
        let shape = view.shape().to_vec();

        if !format::should_quantize(&shape, view.dtype()) {
            ignored.push(name.to_string());
            outputs.insert(
                name.to_string(),
                OutputTensor {
                    dtype: view.dtype(),
                    shape,
                    data: view.data().to_vec(),
                },
            );
            continue;
        }

        let data =
            read_f32(&view, name).with_context(|| format!("reading tensor '{name}' as f32"))?;
        let quantized = format::quantize_tensor(name, &data, &shape, scheme, granularity)
            .with_context(|| format!("quantizing tensor '{name}'"))?;

        if matches!(scheme, Scheme::Int4 { .. }) {
            packed_int4.insert(
                name.to_string(),
                PackedTensorInfo {
                    logical_shape: shape.clone(),
                    elements: shape.iter().product(),
                },
            );
        }

        outputs.extend(quantized);
        quantized_count += 1;
    }

    let ignored_count = ignored.len();
    let config = format::build_config(scheme, granularity, ignored, packed_int4);
    let metadata = format::config_to_metadata(&config)?;

    write_safetensors(&args.output, &outputs, metadata)
        .with_context(|| format!("failed to write '{}'", args.output.display()))?;

    println!(
        "Quantized {quantized_count} tensor(s) ({scheme}, {granularity}), left {ignored_count} tensor(s) unquantized -> {output}",
        scheme = scheme.short_name(),
        granularity = granularity.strategy_name(),
        output = args.output.display(),
    );

    Ok(())
}
