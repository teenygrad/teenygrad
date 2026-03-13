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

use crate::{error::Error, torch::Shape};

impl<'a> TryFrom<Shape<'a>> for teeny_core::fxgraph::shape::Shape {
    type Error = Error;

    fn try_from(shape: Shape) -> Result<Self, Self::Error> {
        let shape = shape
            .dims()
            .ok_or_else(|| Error::InvalidBuffer(format!("{shape:?}")))?
            .into_iter()
            .map(|s| s.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(teeny_core::fxgraph::shape::Shape { shape })
    }
}
