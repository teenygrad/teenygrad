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
