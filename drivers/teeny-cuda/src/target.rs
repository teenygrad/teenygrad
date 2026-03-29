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

use std::collections::HashMap;

use derive_more::Display;

use crate::errors::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Display)]
pub enum Capability {
    #[display("sm_60")]
    Sm60,
    #[display("sm_61")]
    Sm61,
    #[display("sm_70")]
    Sm70,
    #[display("sm_72")]
    Sm72,
    #[display("sm_75")]
    Sm75,
    #[display("sm_80")]
    Sm80,
    #[display("sm_86")]
    Sm86,
    #[display("sm_89")]
    Sm89,
    #[display("sm_90")]
    Sm90,
    #[display("sm_100")]
    Sm100,
    #[display("sm_120")]
    Sm120,
}
