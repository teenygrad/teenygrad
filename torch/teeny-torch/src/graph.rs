/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

#![allow(unsafe_code)]
include!(concat!(env!("OUT_DIR"), "/flatbuffers/graph_generated.rs"));

use crate::error::Result;
pub use fxgraph::*;

pub fn deserialize_graph(buffer: &[u8]) -> Result<Graph> {
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
        Ok(graph) => {
            // Additional validation: check if the graph structure is valid
            if let Some(nodes) = graph.nodes() {
                if nodes.len() > 1_000_000 {
                    return Err(crate::error::Error::InvalidBuffer(
                        "Graph has too many nodes".to_string(),
                    ));
                }
            }
            Ok(graph)
        }
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
