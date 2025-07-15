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
        return Err(Error::ParseValueError("No data found".to_string()));
    }

    let shape = (data.len(), data[0].len());
    let data = data.into_iter().flatten().collect();

    Ok(Array2::from_shape_vec(shape, data)?)
}
