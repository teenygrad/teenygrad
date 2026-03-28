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

use sha2::Digest;
use sha2::Sha256;

pub trait Kernel {
    type Args<'a>;

    fn id(&self) -> [u8; 32] {
        let mut hasher = Sha256::default();
        hasher.update(self.source().as_bytes());
        hasher.finalize().into()
    }

    fn name(&self) -> &str;

    fn source(&self) -> &str;
}

pub trait Program<'a, K: Kernel>: Sized {}
