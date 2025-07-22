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

use crate::error::{Error, Result};
use std::{collections::HashMap, fs::File, io::Write};

pub async fn download_file(
    name: &str,
    url: &str,
    path: &str,
    headers: Option<HashMap<String, String>>,
    show_progress: bool,
) -> Result<()> {
    let client = reqwest::Client::new();
    let mut request = client.get(url);
    if let Some(headers) = headers {
        for (key, value) in headers {
            request = request.header(key, value);
        }
    }
    let mut response = request.send().await.map_err(Error::RequestFailed)?;
    let total_size = response.content_length().unwrap_or(0);
    let mut file = File::create(path).map_err(Error::FileCreateFailed)?;
    let mut content = Vec::new();

    if show_progress {
        let pb = indicatif::ProgressBar::new(total_size);
        pb.set_style(indicatif::ProgressStyle::default_bar()
          .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
          .unwrap()
          .progress_chars("#>-"));
        pb.set_message(name.to_string());

        while let Some(chunk) = response.chunk().await.map_err(Error::RequestFailed)? {
            content.extend_from_slice(&chunk);
            pb.inc(chunk.len() as u64);
        }
        pb.finish_with_message("Download complete");
    } else {
        content = response
            .bytes()
            .await
            .map_err(Error::RequestFailed)?
            .to_vec();
    }

    file.write_all(&content).map_err(Error::FileWriteFailed)?;

    Ok(())
}
