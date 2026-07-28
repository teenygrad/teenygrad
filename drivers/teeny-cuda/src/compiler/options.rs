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

use derive_builder::Builder;
use derive_more::Display;

use crate::compiler::target::Capability;
use crate::errors::{Error, Result};

/// `nvptxcompiler` `--sanitize` value.
#[derive(Debug, Clone, Copy, Display)]
pub enum Sanitizer {
    /// Enable `memcheck`-style memory error detection.
    #[display("memcheck")]
    MemCheck,
}

/// `nvptxcompiler` `--opt-level` value.
#[derive(Debug, Clone, Copy, Display)]
pub enum OptLevel {
    /// No optimization.
    #[display("0")]
    O0,
    /// Light optimization.
    #[display("1")]
    O1,
    /// Default optimization.
    #[display("2")]
    O2,
    /// Aggressive optimization.
    #[display("3")]
    O3,
}

/// `nvptxcompiler` compile options, translated to CLI flags by [`Options::to_compile_options`].
/// Each field corresponds 1:1 to an `nvptxcompiler`/`teenyc` flag of the same name (with `_`
/// replaced by `-`) — see [`Options::parse`] for the string-based `--options` CLI encoding.
#[derive(Builder)]
pub struct Options {
    /// `--allow-expensive-optimizations`.
    #[builder(default = "false")]
    pub allow_expensive_optimizations: bool,

    /// `--compile-as-tools-patch`.
    #[builder(default = "false")]
    pub compile_as_tools_patch: bool,

    /// `--compile-only`.
    #[builder(default = "false")]
    pub compile_only: bool,

    /// `--def-load-cache`.
    #[builder(default = "false")]
    pub def_load_cache: bool,

    /// `--def-store-cache`.
    #[builder(default = "false")]
    pub def_store_cache: bool,

    /// `--device-debug`.
    #[builder(default = "false")]
    pub device_debug: bool,

    /// `--device-function-maxrregcount`.
    #[builder(default = "None")]
    pub device_function_maxrregcount: Option<u32>,

    /// `--disable-optimizer-constants`.
    #[builder(default = "false")]
    pub disable_optimizer_constants: bool,

    /// `--disable-warnings`.
    #[builder(default = "false")]
    pub disable_warnings: bool,

    /// `--dont-merge-basicblocks`.
    #[builder(default = "false")]
    pub dont_merge_basicblocks: bool,

    /// `--entry`: the kernel entry point name.
    #[builder(default = "String::from(\"entry_point\")")]
    pub entry: String,

    /// `--extensible-whole-program`.
    #[builder(default = "false")]
    pub extensible_whole_program: bool,

    /// `--fmad`: enable fused multiply-add contraction.
    #[builder(default = "false")]
    pub fmad: bool,

    /// `--force-load-cache`.
    #[builder(default = "false")]
    pub force_load_cache: bool,

    /// `--force-store-cache`.
    #[builder(default = "false")]
    pub force_store_cache: bool,

    /// `--generate-line-info`.
    #[builder(default = "false")]
    pub generate_line_info: bool,

    /// `--gpu-name`: the target GPU's compute capability.
    #[builder]
    pub gpu_name: Capability,

    /// Explicit PTX ISA version to request from `teenyc` (e.g. `82` for
    /// `8.2`), encoded as `major*10 + minor`. Not an `nvptxcompiler` flag —
    /// excluded from [`Options::to_compile_options`]; consumed separately by
    /// [`crate::compiler::aot::compile_graph`] to override `teenyc`'s
    /// capability-based default when the deployment target's exact CUDA
    /// version is known (e.g. `ptx-version=82` for a Jetson Orin Nano on
    /// CUDA 12.2, since sm_87's own default floor is conservative).
    #[builder(default = "None")]
    pub ptx_version: Option<u32>,

    /// `--maxrregcount` (alias `maxnreg` in [`Options::parse`]'s string encoding).
    #[builder(default = "None")]
    pub maxrregcount: Option<u32>,

    /// `--opt-level`.
    #[builder(default = "None")]
    pub opt_level: Option<OptLevel>,

    /// `--position-independent-code`.
    #[builder(default = "false")]
    pub position_independent_code: bool,

    /// `--preserve-relocs`.
    #[builder(default = "false")]
    pub preserve_relocs: bool,

    /// `--return-at-end`.
    #[builder(default = "false")]
    pub return_at_end: bool,

    /// `--sanitize`.
    #[builder(default = "None")]
    pub sanitize: Option<Sanitizer>,

    /// `--suppress-async-bulk-multicast-advisory-warning`.
    #[builder(default = "false")]
    pub suppress_async_bulk_multicast_advisory_warning: bool,

    /// `--suppress-stack-size-warning`.
    #[builder(default = "false")]
    pub suppress_stack_size_warning: bool,

    /// `--verbose`.
    #[builder(default = "false")]
    pub verbose: bool,

    /// `--warn-on-double-precision-use`.
    #[builder(default = "false")]
    pub warn_on_double_precision_use: bool,

    /// `--warn-on-local-memory-usage`.
    #[builder(default = "false")]
    pub warn_on_local_memory_usage: bool,

    /// `--warn-on-spills`.
    #[builder(default = "false")]
    pub warn_on_spills: bool,

    /// `--warning-as-error`.
    #[builder(default = "false")]
    pub warning_as_error: bool,

    /// `--maxntid`.
    #[builder(default = "None")]
    pub maxntid: Option<u32>,

    /// `--minnctapersm`.
    #[builder(default = "None")]
    pub minnctapersm: Option<u32>,

    /// `--override-directive-values`.
    #[builder(default = "false")]
    pub override_directive_values: bool,

    /// `--make-errors-visible-at-exit`.
    #[builder(default = "false")]
    pub make_errors_visible_at_exit: bool,

    /// `--oFast-compile`.
    #[builder(default = "None")]
    pub ofast_compile: Option<u32>,

    /// `--device-stack-protector`.
    #[builder(default = "false")]
    pub device_stack_protector: bool,

    /// `--g-tensor-memory-access-check`.
    #[builder(default = "false")]
    pub g_tensor_memory_access_check: bool,

    /// `--gno-tensor-memory-access-check`.
    #[builder(default = "false")]
    pub gno_tensor_memory_access_check: bool,

    /// `--split-compile`.
    #[builder(default = "None")]
    pub split_compile: Option<u32>,
}

impl Options {
    /// Renders these options as `nvptxcompiler`/`teenyc` CLI flags.
    pub fn to_compile_options(&self) -> Vec<String> {
        let mut args: Vec<String> = Vec::new();

        if self.allow_expensive_optimizations {
            args.push(String::from("--allow-expensive-optimizations"));
        }

        if self.compile_as_tools_patch {
            args.push(String::from("--compile-as-tools-patch"));
        }

        if self.compile_only {
            args.push(String::from("--compile-only"));
        }

        if self.def_load_cache {
            args.push(String::from("--def-load-cache"));
        }

        if self.def_store_cache {
            args.push(String::from("--def-store-cache"));
        }

        if self.device_debug {
            args.push(String::from("--device-debug"));
        }

        if let Some(device_function_maxrregcount) = self.device_function_maxrregcount {
            args.push(format!(
                "--device-function-maxrregcount={}",
                device_function_maxrregcount
            ));
        }

        if self.disable_optimizer_constants {
            args.push(String::from("--disable-optimizer-constants"));
        }

        if self.disable_warnings {
            args.push(String::from("--disable-warnings"));
        }

        if self.dont_merge_basicblocks {
            args.push(String::from("--dont-merge-basicblocks"));
        }

        args.push(format!("--entry={}", self.entry));

        if self.extensible_whole_program {
            args.push(String::from("--extensible-whole-program"));
        }

        if self.fmad {
            args.push(String::from("--fmad"));
        }

        if self.force_load_cache {
            args.push(String::from("--force-load-cache"));
        }

        if self.force_store_cache {
            args.push(String::from("--force-store-cache"));
        }

        if self.generate_line_info {
            args.push(String::from("--generate-line-info"));
        }

        args.push(format!("--gpu-name={}", self.gpu_name));

        if let Some(maxrregcount) = self.maxrregcount {
            args.push(format!("--maxrregcount={}", maxrregcount));
        }

        if let Some(opt_level) = self.opt_level {
            args.push(format!("--opt-level={}", opt_level));
        }

        if self.position_independent_code {
            args.push(String::from("--position-independent-code"));
        }

        if self.preserve_relocs {
            args.push(String::from("--preserve-relocs"));
        }

        if self.return_at_end {
            args.push(String::from("--return-at-end"));
        }

        if let Some(sanitize) = self.sanitize {
            args.push(format!("--sanitize={}", sanitize));
        }

        if self.suppress_async_bulk_multicast_advisory_warning {
            args.push(String::from(
                "--suppress-async-bulk-multicast-advisory-warning",
            ));
        }

        if self.suppress_stack_size_warning {
            args.push(String::from("--suppress-stack-size-warning"));
        }

        if self.verbose {
            args.push(String::from("--verbose"));
        }

        if self.warn_on_double_precision_use {
            args.push(String::from("--warn-on-double-precision-use"));
        }

        if self.warn_on_local_memory_usage {
            args.push(String::from("--warn-on-local-memory-usage"));
        }

        if self.warn_on_spills {
            args.push(String::from("--warn-on-spills"));
        }

        if self.warning_as_error {
            args.push(String::from("--warning-as-error"));
        }

        if let Some(maxntid) = self.maxntid {
            args.push(format!("--maxntid={}", maxntid));
        }

        if let Some(minnctapersm) = self.minnctapersm {
            args.push(format!("--minnctapersm={}", minnctapersm));
        }

        if self.override_directive_values {
            args.push(String::from("--override-directive-values"));
        }

        if self.make_errors_visible_at_exit {
            args.push(String::from("--make-errors-visible-at-exit"));
        }

        if let Some(ofast_compile) = self.ofast_compile {
            args.push(format!("--oFast-compile={}", ofast_compile));
        }

        if self.device_stack_protector {
            args.push(String::from("--device-stack-protector"));
        }

        if self.g_tensor_memory_access_check {
            args.push(String::from("--g-tensor-memory-access-check"));
        }

        if self.gno_tensor_memory_access_check {
            args.push(String::from("--gno-tensor-memory-access-check"));
        }

        if let Some(split_compile) = self.split_compile {
            args.push(format!("--split-compile={}", split_compile));
        }

        args
    }
}

impl Options {
    /// Parse a comma-separated `key=value` string (as passed via `--options` on
    /// the AOT compile CLI, e.g. `"capability=sm_90,maxnreg=16"`) into `Options`.
    ///
    /// `capability` (alias `gpu-name`) is required. Boolean flags may be given
    /// bare (`key`, meaning `true`) or as `key=true`/`key=false`. Unknown keys
    /// are rejected outright rather than silently ignored, so typos and
    /// not-yet-supported knobs (e.g. a shared-memory limit) surface immediately.
    pub fn parse(input: &str) -> Result<Options> {
        let mut builder = OptionsBuilder::default();
        let mut capability: Option<Capability> = None;

        for pair in input.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (raw_key, value) = match pair.split_once('=') {
                Some((k, v)) => (k.trim(), Some(v.trim())),
                None => (pair, None),
            };
            let key = raw_key.to_ascii_lowercase().replace('_', "-");

            match key.as_str() {
                "capability" | "gpu-name" => {
                    let v = require_value(input, &key, value)?;
                    capability =
                        Some(
                            v.parse::<Capability>()
                                .map_err(|reason| Error::InvalidOptions {
                                    input: input.to_string(),
                                    reason,
                                })?,
                        );
                }
                "ptx-version" => {
                    builder.ptx_version(Some(parse_u32(input, &key, value)?));
                }
                "allow-expensive-optimizations" => {
                    builder.allow_expensive_optimizations(parse_bool(input, &key, value)?);
                }
                "compile-as-tools-patch" => {
                    builder.compile_as_tools_patch(parse_bool(input, &key, value)?);
                }
                "compile-only" => {
                    builder.compile_only(parse_bool(input, &key, value)?);
                }
                "def-load-cache" => {
                    builder.def_load_cache(parse_bool(input, &key, value)?);
                }
                "def-store-cache" => {
                    builder.def_store_cache(parse_bool(input, &key, value)?);
                }
                "device-debug" => {
                    builder.device_debug(parse_bool(input, &key, value)?);
                }
                "device-function-maxrregcount" => {
                    builder.device_function_maxrregcount(Some(parse_u32(input, &key, value)?));
                }
                "disable-optimizer-constants" => {
                    builder.disable_optimizer_constants(parse_bool(input, &key, value)?);
                }
                "disable-warnings" => {
                    builder.disable_warnings(parse_bool(input, &key, value)?);
                }
                "dont-merge-basicblocks" => {
                    builder.dont_merge_basicblocks(parse_bool(input, &key, value)?);
                }
                "entry" => {
                    builder.entry(require_value(input, &key, value)?.to_string());
                }
                "extensible-whole-program" => {
                    builder.extensible_whole_program(parse_bool(input, &key, value)?);
                }
                "fmad" => {
                    builder.fmad(parse_bool(input, &key, value)?);
                }
                "force-load-cache" => {
                    builder.force_load_cache(parse_bool(input, &key, value)?);
                }
                "force-store-cache" => {
                    builder.force_store_cache(parse_bool(input, &key, value)?);
                }
                "generate-line-info" => {
                    builder.generate_line_info(parse_bool(input, &key, value)?);
                }
                "maxnreg" | "maxrregcount" => {
                    builder.maxrregcount(Some(parse_u32(input, &key, value)?));
                }
                "opt-level" => {
                    builder.opt_level(Some(parse_opt_level(input, &key, value)?));
                }
                "position-independent-code" => {
                    builder.position_independent_code(parse_bool(input, &key, value)?);
                }
                "preserve-relocs" => {
                    builder.preserve_relocs(parse_bool(input, &key, value)?);
                }
                "return-at-end" => {
                    builder.return_at_end(parse_bool(input, &key, value)?);
                }
                "sanitize" => {
                    builder.sanitize(Some(parse_sanitizer(input, &key, value)?));
                }
                "suppress-async-bulk-multicast-advisory-warning" => {
                    builder.suppress_async_bulk_multicast_advisory_warning(parse_bool(
                        input, &key, value,
                    )?);
                }
                "suppress-stack-size-warning" => {
                    builder.suppress_stack_size_warning(parse_bool(input, &key, value)?);
                }
                "verbose" => {
                    builder.verbose(parse_bool(input, &key, value)?);
                }
                "warn-on-double-precision-use" => {
                    builder.warn_on_double_precision_use(parse_bool(input, &key, value)?);
                }
                "warn-on-local-memory-usage" => {
                    builder.warn_on_local_memory_usage(parse_bool(input, &key, value)?);
                }
                "warn-on-spills" => {
                    builder.warn_on_spills(parse_bool(input, &key, value)?);
                }
                "warning-as-error" => {
                    builder.warning_as_error(parse_bool(input, &key, value)?);
                }
                "maxntid" => {
                    builder.maxntid(Some(parse_u32(input, &key, value)?));
                }
                "minnctapersm" => {
                    builder.minnctapersm(Some(parse_u32(input, &key, value)?));
                }
                "override-directive-values" => {
                    builder.override_directive_values(parse_bool(input, &key, value)?);
                }
                "make-errors-visible-at-exit" => {
                    builder.make_errors_visible_at_exit(parse_bool(input, &key, value)?);
                }
                "ofast-compile" => {
                    builder.ofast_compile(Some(parse_u32(input, &key, value)?));
                }
                "device-stack-protector" => {
                    builder.device_stack_protector(parse_bool(input, &key, value)?);
                }
                "g-tensor-memory-access-check" => {
                    builder.g_tensor_memory_access_check(parse_bool(input, &key, value)?);
                }
                "gno-tensor-memory-access-check" => {
                    builder.gno_tensor_memory_access_check(parse_bool(input, &key, value)?);
                }
                "split-compile" => {
                    builder.split_compile(Some(parse_u32(input, &key, value)?));
                }
                other => {
                    return Err(Error::InvalidOptions {
                        input: input.to_string(),
                        reason: format!("unknown option key '{other}'"),
                    }
                    .into());
                }
            }
        }

        let capability = capability.ok_or_else(|| Error::InvalidOptions {
            input: input.to_string(),
            reason: "missing required 'capability' key, e.g. capability=sm_90".to_string(),
        })?;
        builder.gpu_name(capability);

        builder.build().map_err(|e| {
            Error::InvalidOptions {
                input: input.to_string(),
                reason: e.to_string(),
            }
            .into()
        })
    }
}

fn require_value<'a>(input: &str, key: &str, value: Option<&'a str>) -> Result<&'a str> {
    value.ok_or_else(|| {
        Error::InvalidOptions {
            input: input.to_string(),
            reason: format!("'{key}' requires a value, e.g. {key}=<value>"),
        }
        .into()
    })
}

fn parse_bool(input: &str, key: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None => Ok(true),
        Some(v) => match v.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" => Ok(false),
            other => Err(Error::InvalidOptions {
                input: input.to_string(),
                reason: format!("invalid boolean value '{other}' for '{key}'"),
            }
            .into()),
        },
    }
}

fn parse_u32(input: &str, key: &str, value: Option<&str>) -> Result<u32> {
    let v = require_value(input, key, value)?;
    v.parse::<u32>().map_err(|_| {
        Error::InvalidOptions {
            input: input.to_string(),
            reason: format!("invalid integer value '{v}' for '{key}'"),
        }
        .into()
    })
}

fn parse_opt_level(input: &str, key: &str, value: Option<&str>) -> Result<OptLevel> {
    let v = require_value(input, key, value)?;
    match v.to_ascii_lowercase().as_str() {
        "0" | "o0" => Ok(OptLevel::O0),
        "1" | "o1" => Ok(OptLevel::O1),
        "2" | "o2" => Ok(OptLevel::O2),
        "3" | "o3" => Ok(OptLevel::O3),
        other => Err(Error::InvalidOptions {
            input: input.to_string(),
            reason: format!("invalid opt-level '{other}'; expected one of 0, 1, 2, 3"),
        }
        .into()),
    }
}

fn parse_sanitizer(input: &str, key: &str, value: Option<&str>) -> Result<Sanitizer> {
    let v = require_value(input, key, value)?;
    match v.to_ascii_lowercase().as_str() {
        "memcheck" => Ok(Sanitizer::MemCheck),
        other => Err(Error::InvalidOptions {
            input: input.to_string(),
            reason: format!("invalid sanitize value '{other}'; expected 'memcheck'"),
        }
        .into()),
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn parses_capability_and_maxnreg_alias() {
        let opts = Options::parse("capability=sm_90,maxnreg=16").unwrap();
        assert_eq!(opts.gpu_name, Capability::Sm90);
        assert_eq!(opts.maxrregcount, Some(16));
    }

    #[test]
    fn bare_bool_flag_means_true() {
        let opts = Options::parse("capability=sm_90,verbose").unwrap();
        assert!(opts.verbose);
    }

    #[test]
    fn missing_capability_errors() {
        assert!(Options::parse("maxnreg=16").is_err());
    }

    #[test]
    fn unknown_key_errors() {
        assert!(Options::parse("capability=sm_90,shared-memory=25k").is_err());
    }

    #[test]
    fn parses_ptx_version_override() {
        let opts = Options::parse("capability=sm_87,ptx-version=82").unwrap();
        assert_eq!(opts.gpu_name, Capability::Sm87);
        assert_eq!(opts.ptx_version, Some(82));
    }

    #[test]
    fn ptx_version_defaults_to_none() {
        let opts = Options::parse("capability=sm_87").unwrap();
        assert_eq!(opts.ptx_version, None);
    }
}
