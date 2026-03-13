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

#![allow(unsafe_op_in_unsafe_fn)]
#![allow(unsafe_code)]
#![allow(unused_imports)]
include!(concat!(env!("OUT_DIR"), "/flatbuffers/graph_generated.rs"));

use crate::error::Result;
pub use fxgraph::*;

pub fn deserialize_graph<'a>(buffer: &'a [u8]) -> Result<Graph<'a>> {
    // Validate buffer size before deserialization
    if buffer.is_empty() {
        return Err(crate::error::Error::InvalidBuffer(
            "Buffer is empty".to_string(),
        ));
    }

    // Check if buffer is too large (safety check)
    if buffer.len() > 100_000_000 {
        // 100MB limit
        return Err(crate::error::Error::InvalidBuffer(format!(
            "Buffer too large: {} bytes",
            buffer.len()
        )));
    }

    // Additional buffer integrity checks
    if buffer.len() < 8 {
        return Err(crate::error::Error::InvalidBuffer(
            "Buffer too small to contain valid flatbuffer".to_string(),
        ));
    }

    // Try to deserialize with better error context and multiple fallback strategies
    let result = flatbuffers::root::<Graph>(buffer);

    match result {
        Ok(graph) => Ok(graph),
        Err(e) => {
            // Try to provide more specific error information
            let error_msg = format!(
                "Failed to deserialize graph: {} (buffer size: {} bytes, error type: {:?})",
                e,
                buffer.len(),
                std::any::type_name::<flatbuffers::InvalidFlatbuffer>()
            );

            // Check if this is a specific range error
            if error_msg.contains("Range") && error_msg.contains("out of bounds") {
                return Err(crate::error::Error::DeserializationFailed(format!(
                    "Buffer corruption detected - range out of bounds. This usually indicates \
                     a mismatch between Python serialization and Rust deserialization. {error_msg}"
                )));
            }

            Err(crate::error::Error::DeserializationFailed(error_msg))
        }
    }
}
