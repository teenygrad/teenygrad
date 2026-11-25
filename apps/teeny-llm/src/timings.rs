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

//! [`Timings`]: how long a generation took, phase by phase, as the model measured it, and
//! the [`Recorder`] models measure it with.
//!
//! A generation has two phases. The *prompt* phase runs the prompt's new tokens (those not
//! reused from a KV cache) and yields the first token. The *generation* phase runs one
//! decode step per further token. Prompt throughput and decode throughput are the two
//! numbers a model's speed comes down to; the per-step latencies give decode's spread.

use std::time::{Duration, Instant};

/// How long one generation took, phase by phase.
///
/// The first generated token comes out of the prompt phase, so the generation phase's
/// rates are over its decode steps: `generated_tokens - 1` of them, one per entry of
/// [`token_latencies`](Self::token_latencies).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Timings {
    /// From the request being accepted to its prompt starting to run: waiting for the
    /// model (busy with earlier requests) and preparing the prompt.
    pub queued: Duration,
    /// Prompt tokens the model ran.
    pub prompt_tokens: u32,
    /// Prompt tokens reused from the KV cache (a prefix shared with an earlier request),
    /// which were not run.
    pub cached_tokens: u32,
    /// Running the prompt tokens, up to the first token's logits.
    pub prompt: Duration,
    /// From the request being accepted to its first token: what a client waits before
    /// output starts.
    pub first_token: Duration,
    /// Tokens generated, the first included.
    pub generated_tokens: u32,
    /// From the first token to the last.
    pub generation: Duration,
    /// Each decode step's latency, in order: the time from one token to the next.
    pub token_latencies: Vec<Duration>,
}

impl Timings {
    /// Prompt tokens run per second (0 when none ran).
    pub fn prompt_tokens_per_second(&self) -> f64 {
        rate(self.prompt_tokens as usize, self.prompt)
    }

    /// Decode steps per second over the generation phase (0 with fewer than two tokens).
    pub fn generation_tokens_per_second(&self) -> f64 {
        rate(self.token_latencies.len(), self.generation)
    }

    /// The `p`th percentile (0 to 100) of the decode step latencies, nearest-rank; `None`
    /// with no decode steps.
    pub fn latency_percentile(&self, p: f64) -> Option<Duration> {
        percentile(&self.token_latencies, p)
    }

    /// From the request being accepted to its last token.
    pub fn total(&self) -> Duration {
        self.first_token + self.generation
    }
}

fn rate(n: usize, d: Duration) -> f64 {
    if n == 0 || d.is_zero() {
        0.0
    } else {
        n as f64 / d.as_secs_f64()
    }
}

/// The `p`th percentile (0 to 100) of `values`, nearest-rank; `None` when empty.
pub fn percentile(values: &[Duration], p: f64) -> Option<Duration> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = (p.clamp(0.0, 100.0) / 100.0 * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

/// Measures a generation as it runs, into [`Timings`].
///
/// Make one when the request is accepted, call [`begin_prompt`](Self::begin_prompt) just
/// before running the prompt, [`end_prompt`](Self::end_prompt) once its logits are back,
/// [`token`](Self::token) as each token is chosen, and [`finish`](Self::finish) at the end.
#[derive(Debug, Clone)]
pub struct Recorder {
    accepted: Instant,
    prompt_started: Option<Instant>,
    first_token: Option<Instant>,
    last_token: Option<Instant>,
    timings: Timings,
}

impl Recorder {
    /// Starts measuring a request accepted now.
    pub fn new() -> Self {
        Self::accepted_at(Instant::now())
    }

    /// Starts measuring a request accepted at `accepted` (when it was queued).
    pub fn accepted_at(accepted: Instant) -> Self {
        Self {
            accepted,
            prompt_started: None,
            first_token: None,
            last_token: None,
            timings: Timings::default(),
        }
    }

    /// The prompt starts running now.
    pub fn begin_prompt(&mut self) {
        let now = Instant::now();
        self.timings.queued = now - self.accepted;
        self.prompt_started = Some(now);
    }

    /// The prompt has run: `run` tokens of it, after `cached` reused from a cache.
    pub fn end_prompt(&mut self, run: usize, cached: usize) {
        let now = Instant::now();
        let started = *self.prompt_started.get_or_insert(self.accepted);
        self.timings.prompt = now - started;
        self.timings.prompt_tokens = count(run);
        self.timings.cached_tokens = count(cached);
    }

    /// A token was chosen now.
    pub fn token(&mut self) {
        let now = Instant::now();
        match self.last_token {
            None => {
                self.first_token = Some(now);
                self.timings.first_token = now - self.accepted;
            }
            Some(last) => self.timings.token_latencies.push(now - last),
        }
        self.last_token = Some(now);
        self.timings.generated_tokens = self.timings.generated_tokens.saturating_add(1);
    }

    /// What was measured.
    pub fn finish(mut self) -> Timings {
        if let (Some(first), Some(last)) = (self.first_token, self.last_token) {
            self.timings.generation = last - first;
        }
        self.timings
    }
}

impl Default for Recorder {
    fn default() -> Self {
        Self::new()
    }
}

/// Timings for a model that produced its whole output at once (the echo models), accepted
/// at `accepted`: `prompt` tokens run and `generated` produced, measured as they are
/// counted, so the shape is real and the durations tiny.
pub(crate) fn instant(accepted: Instant, prompt: u32, generated: u32) -> Timings {
    let mut r = Recorder::accepted_at(accepted);
    r.begin_prompt();
    r.end_prompt(prompt as usize, 0);
    for _ in 0..generated {
        r.token();
    }
    r.finish()
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    #[test]
    fn test_rates_are_over_the_phases_own_tokens() {
        let t = Timings {
            prompt_tokens: 64,
            prompt: ms(500),
            generated_tokens: 11,
            generation: ms(1000),
            token_latencies: vec![ms(100); 10],
            ..Timings::default()
        };
        assert_eq!(t.prompt_tokens_per_second(), 128.0);
        assert_eq!(t.generation_tokens_per_second(), 10.0);
        assert_eq!(Timings::default().prompt_tokens_per_second(), 0.0);
        assert_eq!(Timings::default().generation_tokens_per_second(), 0.0);
    }

    #[test]
    fn test_percentiles_are_nearest_rank() {
        let values: Vec<Duration> = (1..=10).rev().map(ms).collect();
        assert_eq!(percentile(&values, 50.0), Some(ms(5)));
        assert_eq!(percentile(&values, 90.0), Some(ms(9)));
        assert_eq!(percentile(&values, 99.0), Some(ms(10)));
        assert_eq!(percentile(&values, 0.0), Some(ms(1)));
        assert_eq!(percentile(&[], 50.0), None);
    }

    #[test]
    fn test_recorder_splits_a_generation_into_its_phases() {
        let accepted = Instant::now();
        let mut r = Recorder::accepted_at(accepted);
        std::thread::sleep(ms(2));
        r.begin_prompt();
        std::thread::sleep(ms(2));
        r.end_prompt(5, 3);
        for _ in 0..4 {
            r.token();
            std::thread::sleep(ms(1));
        }
        let t = r.finish();
        assert!(t.queued >= ms(2) && t.prompt >= ms(2), "{t:?}");
        assert_eq!((t.prompt_tokens, t.cached_tokens), (5, 3));
        assert!(t.first_token >= t.queued + t.prompt);
        assert_eq!(t.generated_tokens, 4);
        assert_eq!(t.token_latencies.len(), 3);
        assert_eq!(t.generation, t.token_latencies.iter().sum());
    }
}
