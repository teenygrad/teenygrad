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

use std::marker::PhantomData;

use teeny_core::{context::buffer::Buffer, dtype::Dtype};

pub struct CudaBuffer<'a, D: Dtype> {
    _data: PhantomData<D>,
    _unused: PhantomData<&'a ()>,
}

impl<'a, D: Dtype> Buffer<'a, D> for CudaBuffer<'a, D> {}
