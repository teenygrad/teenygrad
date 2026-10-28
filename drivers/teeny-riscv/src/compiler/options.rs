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

use teeny_compiler::compiler::backend::llvm::compiler::LogLevel;

use crate::compiler::target::Capability;
use crate::errors::{Error, Result};

/// RISC-V AOT compile options, parsed from the AOT compile CLI's `--options` string (see
/// [`Options::parse`]). The RISC-V counterpart of `teeny_cuda::compiler::options::Options`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    /// Target chip, passed to `teenyc` as `-C target-cpu`.
    pub capability: Capability,
    /// `teenyc` pipeline-stage log level.
    pub log_level: Option<LogLevel>,
}

impl Options {
    /// Parse a comma-separated `key=value` string (as passed via `--options` on the AOT compile
    /// CLI, e.g. `"capability=spacemit-k3"`) into `Options`.
    ///
    /// `capability` is required. Unknown keys are rejected outright rather than silently
    /// ignored, so typos and not-yet-supported knobs surface immediately.
    pub fn parse(input: &str) -> Result<Options> {
        let invalid = |reason: String| Error::InvalidOptions {
            input: input.to_string(),
            reason,
        };
        let mut capability: Option<Capability> = None;
        let mut log_level: Option<LogLevel> = None;

        for pair in input.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (raw_key, value) = match pair.split_once('=') {
                Some((k, v)) => (k.trim(), Some(v.trim())),
                None => (pair, None),
            };
            let key = raw_key.to_ascii_lowercase().replace('_', "-");
            let value = value
                .ok_or_else(|| invalid(format!("'{key}' requires a value, e.g. {key}=<value>")));

            match key.as_str() {
                "capability" => capability = Some(value?.parse::<Capability>().map_err(invalid)?),
                "log-level" => log_level = Some(value?.parse::<LogLevel>().map_err(invalid)?),
                other => return Err(invalid(format!("unknown option '{other}'")).into()),
            }
        }

        let capability = capability.ok_or_else(|| {
            invalid("missing required 'capability' key, e.g. capability=generic-rvv1.0".to_string())
        })?;
        Ok(Options {
            capability,
            log_level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_capability() {
        let opts = Options::parse("capability=spacemit-k3").unwrap();
        assert_eq!(opts.capability, Capability::SpacemitK3);
        assert_eq!(opts.log_level, None);
    }

    #[test]
    fn parses_log_level_and_normalizes_keys() {
        let opts = Options::parse(" Capability = generic-rvv1.0 , log_level=debug ").unwrap();
        assert_eq!(opts.capability, Capability::GenericRvv1_0);
        assert_eq!(opts.log_level, Some(LogLevel::Debug));
    }

    #[test]
    fn requires_capability() {
        assert!(Options::parse("").is_err());
        assert!(Options::parse("log-level=debug").is_err());
    }

    #[test]
    fn rejects_bad_input() {
        assert!(Options::parse("capability=spacemit-k9").is_err());
        assert!(Options::parse("capability").is_err());
        assert!(Options::parse("capability=spacemit-k3,maxnreg=16").is_err());
        assert!(Options::parse("capability=spacemit-k3,log-level=verbose").is_err());
    }
}
