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

use std::str::FromStr;

use ndarray::{Array2, ArrayBase, Ix2, OwnedRepr};

use crate::error::{Error, Result};

pub async fn load_csv<T: FromStr>(
    url: &str,
    delimiter: u8,
) -> Result<ArrayBase<OwnedRepr<T>, Ix2>> {
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .from_reader(body.as_bytes());
    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let record = record
            .iter()
            .map(|s| s.parse::<T>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| Error::ParseValueError(format!("{record:?}")))?;
        data.push(record);
    }

    if data.is_empty() {
        return Err(Error::ParseValueError("No data found".to_string()).into());
    }

    let shape = (data.len(), data[0].len());
    let data = data.into_iter().flatten().collect();

    Ok(Array2::from_shape_vec(shape, data)?)
}
