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

use crate::{
    error::{Error, Result},
    fetch::fetch_content,
};
use std::{fs::File, io::Write, path::Path};

pub async fn download_file(
    name: &str,
    url: &str,
    path: &Path,
    headers: Option<HeaderMap>,
    show_progress: bool,
) -> Result<()> {
    let mut file = File::create(path.join(name)).map_err(Error::FileCreateFailed)?;
    let content = fetch_content(name, url, headers, show_progress).await?;
    file.write_all(&content).map_err(Error::FileWriteFailed)?;

    Ok(())
}
