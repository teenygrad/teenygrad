/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use alloc::{boxed::Box, vec::Vec};
use async_trait::async_trait;
use core::pin::Pin;
use smol::io::{AsyncRead, AsyncReadExt};

use super::{AsyncTensorRead, error::TensorError};

#[async_trait]
impl AsyncTensorRead<u8> for u8 {
    async fn read_to_vec(
        &mut self,
        source: &mut Pin<&mut (dyn AsyncRead + Send)>,
        buf: &mut [u8],
    ) -> Result<Vec<u8>, TensorError> {
        let mut result = Vec::new();

        loop {
            let n = source.read(buf).await?;
            if n == 0 {
                break;
            }

            result.extend_from_slice(&buf[..n]);
        }

        Ok(result)
    }
}
