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

pub mod compiler;
pub mod device;
pub mod errors;
pub mod model;
pub mod runtime;
pub mod testing;

mod cuda;

/// Signal nsys (or any CUDA profiler) to start capturing.
///
/// # Safety
/// Calls `cudaProfilerStart` via the CUDA runtime. Safe to call multiple times;
/// has no effect if no profiler is attached.
pub unsafe fn cuda_profiler_start() {
    unsafe { cuda::cudaProfilerStart() };
}

/// Signal nsys (or any CUDA profiler) to stop capturing.
///
/// # Safety
/// Calls `cudaProfilerStop` via the CUDA runtime.
pub unsafe fn cuda_profiler_stop() {
    unsafe { cuda::cudaProfilerStop() };
}
