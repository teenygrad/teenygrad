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

use thiserror::Error;

pub type Result<T> = anyhow::Result<T>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Request failed: {0}")]
    RequestFailed(reqwest::Error),

    #[error("File create failed: {0}")]
    FileCreateFailed(std::io::Error),

    #[error("File write failed: {0}")]
    FileWriteFailed(std::io::Error),

    #[error("HTTP error: {0}")]
    HttpError(reqwest::StatusCode),
}
