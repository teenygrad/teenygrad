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

use std::fmt;

use derive_builder::Builder;
use derive_more::Display;

use crate::target::Capability;

#[derive(Debug, Clone, Copy, Display)]
pub enum Sanitizer {
    #[display("memcheck")]
    MemCheck,
}

#[derive(Debug, Clone, Copy, Display)]
pub enum OptLevel {
    #[display("0")]
    O0,
    #[display("1")]
    O1,
    #[display("2")]
    O2,
    #[display("3")]
    O3,
}

#[derive(Builder)]
pub struct Options {
    #[builder(default = "false")]
    pub allow_expensive_optimizations: bool,

    #[builder(default = "false")]
    pub compile_as_tools_patch: bool,

    #[builder(default = "false")]
    pub compile_only: bool,

    #[builder(default = "false")]
    pub def_load_cache: bool,

    #[builder(default = "false")]
    pub def_store_cache: bool,

    #[builder(default = "false")]
    pub device_debug: bool,

    #[builder]
    pub device_function_maxrregcount: Option<i32>,

    #[builder(default = "false")]
    pub disable_optimizer_constants: bool,

    #[builder(default = "false")]
    pub disable_warnings: bool,

    #[builder(default = "false")]
    pub dont_merge_basicblocks: bool,

    #[builder(default = "String::from(\"entry\")")]
    pub entry: String,

    #[builder(default = "false")]
    pub extensible_whole_program: bool,

    #[builder(default = "true")]
    pub fmad: bool,

    #[builder(default = "false")]
    pub force_load_cache: bool,

    #[builder(default = "false")]
    pub force_store_cache: bool,

    #[builder(default = "false")]
    pub generate_line_info: bool,

    #[builder]
    pub gpu_name: Capability,

    #[builder]
    pub maxrregcount: Option<i32>,

    #[builder]
    pub opt_level: Option<OptLevel>,

    #[builder(default = "true")]
    pub position_independent_code: bool,

    #[builder(default = "false")]
    pub preserve_relocs: bool,

    #[builder(default = "false")]
    pub return_at_end: bool,

    #[builder]
    pub sanitize: Option<Sanitizer>,

    #[builder(default = "false")]
    pub suppress_async_bulk_multicast_advisory_warning: bool,

    #[builder(default = "false")]
    pub suppress_stack_size_warning: bool,

    #[builder(default = "false")]
    pub verbose: bool,

    #[builder(default = "false")]
    pub warn_on_double_precision_use: bool,

    #[builder(default = "false")]
    pub warn_on_local_memory_usage: bool,

    #[builder(default = "true")]
    pub warn_on_spills: bool,

    #[builder(default = "false")]
    pub warning_as_error: bool,

    #[builder]
    pub maxntid: Option<i32>,

    #[builder]
    pub minnctapersm: Option<i32>,

    #[builder(default = "false")]
    pub override_directive_values: bool,

    #[builder(default = "false")]
    pub make_errors_visible_at_exit: bool,

    #[builder]
    pub ofast_compile: Option<i32>,

    #[builder(default = "false")]
    pub device_stack_protector: bool,

    #[builder(default = "false")]
    pub g_tensor_memory_access_check: bool,

    #[builder(default = "false")]
    pub gno_tensor_memory_access_check: bool,

    #[builder]
    pub split_compile: Option<i32>,
}

impl Options {
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
                "--device-function-maxrregcount {}",
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

        args.push(format!("--entry {}", self.entry));

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

        args.push(format!("--gpu-name {}", self.gpu_name));

        if let Some(maxrregcount) = self.maxrregcount {
            args.push(format!("--maxrregcount {}", maxrregcount));
        }

        if let Some(opt_level) = self.opt_level {
            args.push(format!("--opt-level {}", opt_level));
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
            args.push(format!("--sanitize {}", sanitize));
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
            args.push(format!("--maxntid {}", maxntid));
        }

        if let Some(minnctapersm) = self.minnctapersm {
            args.push(format!("--minnctapersm {}", minnctapersm));
        }

        if self.override_directive_values {
            args.push(String::from("--override-directive-values"));
        }

        if self.make_errors_visible_at_exit {
            args.push(String::from("--make-errors-visible-at-exit"));
        }

        if let Some(ofast_compile) = self.ofast_compile {
            args.push(format!("--oFast-compile {}", ofast_compile));
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
            args.push(format!("--split-compile {}", split_compile));
        }

        args
    }
}
