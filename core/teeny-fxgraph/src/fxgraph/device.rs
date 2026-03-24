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

use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

use crate::errors::Error;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Device {
    Cpu(String),
    Cuda(String),
}

impl Display for Device {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu(s) => write!(f, "cpu:{}", s),
            Device::Cuda(s) => write!(f, "cuda:{}", s),
        }
    }
}

impl FromStr for Device {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("cpu:") {
            Ok(Device::Cpu(s.split_at(4).1.to_string()))
        } else if s.starts_with("cuda:") {
            Ok(Device::Cuda(s.split_at(5).1.to_string()))
        } else {
            Err(Error::InvalidDevice(s.to_string()))
        }
    }
}
