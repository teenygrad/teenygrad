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
use error::TensorError;
use smol::io::AsyncRead;

pub mod error;
pub mod memory;
pub mod types;

pub trait Tensor<T: Clone> {
    fn shape(&self) -> &[i64];
    fn reshape(&mut self, shape: &[i64]) -> Box<dyn Tensor<T>>;

    fn dot(&self, other: &dyn Tensor<T>) -> Box<dyn Tensor<T>>;
    fn relu(&self) -> Box<dyn Tensor<T>>;
    fn log_softmax(&self) -> Box<dyn Tensor<T>>;
}

#[async_trait]
pub trait AsyncTensorRead<T: Clone> {
    async fn read_to_vec(
        &mut self,
        source: &mut Pin<&mut (dyn AsyncRead + Send)>,
        buf: &mut [u8],
    ) -> Result<Vec<T>, TensorError>;
}
