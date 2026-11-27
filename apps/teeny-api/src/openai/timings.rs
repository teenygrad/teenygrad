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

//! `timings`: how long a generation took, in llama.cpp server's format.
//!
//! Not part of OpenAI's API: an extra top-level object on chat and text completion
//! responses (and on a stream's final usage chunk), which OpenAI clients ignore and
//! llama.cpp-aware tools read. The `prompt_*`, `predicted_*` and `cache_n` fields are
//! llama.cpp's; `queued_ms`, `first_token_ms` and `predicted_latencies_ms` are teenygrad's.
//!
//! The first predicted token comes out of the prompt phase, so `predicted_ms`,
//! `predicted_per_token_ms` and `predicted_per_second` are over the decode steps after it
//! (`predicted_n - 1`, one per `predicted_latencies_ms` entry), not over `predicted_n`.

use serde::{Deserialize, Serialize};

/// How long one generation took, phase by phase.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    /// Prompt tokens reused from the KV cache, which were not run.
    pub cache_n: u32,
    /// Prompt tokens run.
    pub prompt_n: u32,
    /// Running them, up to the first token's logits.
    pub prompt_ms: f64,
    pub prompt_per_token_ms: f64,
    pub prompt_per_second: f64,
    /// Tokens generated, the first included.
    pub predicted_n: u32,
    /// From the first generated token to the last.
    pub predicted_ms: f64,
    pub predicted_per_token_ms: f64,
    pub predicted_per_second: f64,
    /// From the request being accepted to its prompt starting to run.
    #[serde(default)]
    pub queued_ms: f64,
    /// From the request being accepted to its first token.
    #[serde(default)]
    pub first_token_ms: f64,
    /// Each decode step's latency, in order.
    #[serde(default)]
    pub predicted_latencies_ms: Vec<f64>,
}

#[cfg(feature = "llm")]
impl From<&teeny_llm::Timings> for Timings {
    fn from(t: &teeny_llm::Timings) -> Self {
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
        let per = |total: f64, n: usize| if n == 0 { 0.0 } else { total / n as f64 };
        let steps = t.token_latencies.len();
        Self {
            cache_n: t.cached_tokens,
            prompt_n: t.prompt_tokens,
            prompt_ms: ms(t.prompt),
            prompt_per_token_ms: per(ms(t.prompt), t.prompt_tokens as usize),
            prompt_per_second: t.prompt_tokens_per_second(),
            predicted_n: t.generated_tokens,
            predicted_ms: ms(t.generation),
            predicted_per_token_ms: per(ms(t.generation), steps),
            predicted_per_second: t.generation_tokens_per_second(),
            queued_ms: ms(t.queued),
            first_token_ms: ms(t.first_token),
            predicted_latencies_ms: t.token_latencies.iter().copied().map(ms).collect(),
        }
    }
}

#[cfg(all(test, feature = "llm"))]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_timings_take_llama_cpp_names_and_rates() {
        let ms = Duration::from_millis;
        let t = Timings::from(&teeny_llm::Timings {
            queued: ms(5),
            prompt_tokens: 64,
            cached_tokens: 16,
            prompt: ms(500),
            first_token: ms(510),
            generated_tokens: 5,
            generation: ms(400),
            token_latencies: vec![ms(100); 4],
        });
        assert_eq!((t.cache_n, t.prompt_n, t.predicted_n), (16, 64, 5));
        assert_eq!(t.prompt_ms, 500.0);
        assert_eq!(t.prompt_per_token_ms, 7.8125);
        assert_eq!(t.prompt_per_second, 128.0);
        assert_eq!(t.predicted_ms, 400.0);
        assert_eq!(t.predicted_per_token_ms, 100.0);
        assert_eq!(t.predicted_per_second, 10.0);
        assert_eq!((t.queued_ms, t.first_token_ms), (5.0, 510.0));
        assert_eq!(t.predicted_latencies_ms, [100.0; 4]);
        let json = serde_json::to_value(&t).unwrap();
        for key in [
            "cache_n",
            "prompt_n",
            "prompt_ms",
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_token_ms",
            "predicted_per_second",
        ] {
            assert!(json.get(key).is_some(), "{key} in {json}");
        }
    }
}
