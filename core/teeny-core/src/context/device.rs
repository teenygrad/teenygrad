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

use crate::{
    context::{
        buffer::Buffer,
        program::{Kernel, Program},
    },
    dtype::Dtype,
    errors::Result,
};

pub trait Device<'a>: Sized {
    type Buffer<D: Dtype>: Buffer<'a, D>;
    type Program<K: Kernel>: Program<'a, K>;

    fn buffer<D: Dtype>(&self, size: &[usize]) -> Result<Self::Buffer<D>>;

    fn compile<K: Kernel>(&self, kernel: &K) -> Result<Self::Program<K>>;
}
