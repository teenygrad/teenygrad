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

use reqwest::header::HeaderMap;

use crate::error::{Error, Result};

pub async fn fetch_content(
    name: &str,
    url: &str,
    headers: Option<HeaderMap>,
    show_progress: bool,
) -> Result<Vec<u8>> {
    let client = reqwest::Client::new();
    let mut request = client.get(url);
    if let Some(headers) = headers {
        request = request.headers(headers);
    }

    let mut response = request.send().await.map_err(Error::RequestFailed)?;
    let total_size = response.content_length().unwrap_or(0);
    let mut content = Vec::new();

    if !response.status().is_success() {
        return Err(Error::HttpError(response.status()).into());
    }

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

    Ok(content)
}
