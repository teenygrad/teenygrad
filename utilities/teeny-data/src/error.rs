/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
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

pub type Result<T> = anyhow::Result<T>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to download CSV: {0}")]
    DownloadError(#[from] reqwest::Error),

    #[error("Failed to parse CSV: {0}")]
    ParseError(#[from] csv::Error),

    #[error("Failed to convert to ndarray: {0}")]
    ArrayError(#[from] ndarray::ShapeError),

    #[error("Failed to parse value: {0}")]
    ParseValueError(String),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(teeny_core::safetensors::SafeTensorsError),
}
