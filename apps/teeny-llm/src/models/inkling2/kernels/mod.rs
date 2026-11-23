/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

//! Kernels specific to Inkling-2, alongside the shared ones in [`crate::kernels`].

pub mod attention;
pub mod block_attn;
pub mod cat2;
pub mod expert_gather;
pub mod gated_mul;
pub mod l2_clip;
pub mod mix;
pub mod norm;
pub mod rel_bias;
pub mod router;
pub mod scatter_add;
pub mod sconv;
pub mod slot_ids;
pub mod softcap;
pub mod swiglu;
