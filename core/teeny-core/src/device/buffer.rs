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

use crate::{dtype::Num, errors::Result};

pub trait Buffer<'a, N: Num>: Sized {
    /// Copy elements from a host slice into this device buffer.
    /// The slice length must not exceed the buffer's allocated element count.
    fn to_device(&mut self, src: &[N]) -> Result<()>;

    /// Copy elements from this device buffer into a host slice.
    /// The slice length must not exceed the buffer's allocated element count.
    fn to_host(&self, dst: &mut [N]) -> Result<()>;
}
