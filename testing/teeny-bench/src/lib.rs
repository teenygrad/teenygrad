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

//! `teeny-bench`: llama-bench-style benchmarks of teenygrad's chat models.
//!
//! A benchmark runs [`Test`]s against a [`Target`] and reports a [`TestResult`] per test:
//!
//! - `pp<N>` runs an N-token prompt and generates one token: prompt processing throughput
//!   and the time to first token;
//! - `tg<N>` generates N tokens after a short prompt: decode throughput and the latency of
//!   each decode step (p50, p90, p99 over every run).
//!
//! Every number comes from the model's own [`Timings`], so a model in-process
//! ([`LocalTarget`]) and one behind teeny-api ([`ApiTarget`], reading the llama.cpp-format
//! `timings` it returns) are measured alike; over the API, the round trip is recorded too.
//!
//! Prompt lengths are calibrated with two probe requests (the chat template adds tokens of
//! its own, and over the API there is no tokenizer to count with), and every request starts
//! with a run tag of its own, so a model's KV cache reuses no more than the template's
//! opening. The tokens actually run are reported beside each test's name.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail, ensure};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};
use teeny_llm::chat::{self, ChatModel, ChatRequest, Message};
use teeny_llm::timings::{Timings, percentile};

/// One request's measurements.
#[derive(Debug, Clone)]
pub struct Sample {
    /// The model's own.
    pub timings: Timings,
    /// From sending the request to having the whole response.
    pub round_trip: Duration,
}

/// Something to benchmark: a chat model, in-process or behind an API.
pub trait Target: Send + Sync {
    /// What is being benchmarked, for the report (`functiongemma3:270m (in-process)`).
    fn describe(&self) -> String;

    /// Asks the model to reply to the user message `text`, generating at most `max_tokens`.
    fn chat(&self, text: String, max_tokens: u32) -> BoxFuture<'_, anyhow::Result<Sample>>;
}

/// A chat model in this process.
pub struct LocalTarget {
    model: Arc<dyn ChatModel>,
    label: String,
}

impl LocalTarget {
    /// Benchmarks `model`; `label` says how it runs (`in-process, cuda`).
    pub fn new(model: Arc<dyn ChatModel>, label: &str) -> Self {
        Self {
            model,
            label: label.to_owned(),
        }
    }
}

impl Target for LocalTarget {
    fn describe(&self) -> String {
        format!("{} ({})", self.model.info().id, self.label)
    }

    fn chat(&self, text: String, max_tokens: u32) -> BoxFuture<'_, anyhow::Result<Sample>> {
        Box::pin(async move {
            let id = self.model.info().id;
            let request = ChatRequest {
                model: id.clone(),
                messages: vec![Message::user(text)],
                max_completion_tokens: Some(max_tokens),
                include_usage: true,
                ..ChatRequest::default()
            };
            chat::validate(&id, &self.model.capabilities(), &request)?;
            let started = Instant::now();
            let done = chat::collect(&id, 1, self.model.chat(request).await?).await?;
            let round_trip = started.elapsed();
            let timings = done
                .timings
                .with_context(|| format!("model `{id}` does not report its timings"))?;
            Ok(Sample {
                timings,
                round_trip,
            })
        })
    }
}

/// A chat model teeny-api serves.
pub struct ApiTarget {
    client: reqwest::Client,
    base: String,
    model: String,
}

impl ApiTarget {
    /// Benchmarks `model` (or the first chat model listed, without one) on the teeny-api at
    /// `base` (`http://127.0.0.1:8000`).
    pub async fn connect(base: &str, model: Option<&str>) -> anyhow::Result<Self> {
        let client = reqwest::Client::new();
        let base = base.trim_end_matches('/').to_owned();
        let model = match model {
            Some(m) => m.to_owned(),
            None => {
                let url = format!("{base}/v1/models");
                let list: serde_json::Value = client
                    .get(&url)
                    .send()
                    .await
                    .with_context(|| format!("listing the models at {url}; is teeny-api running?"))?
                    .error_for_status()?
                    .json()
                    .await?;
                list["data"][0]["id"]
                    .as_str()
                    .with_context(|| format!("{url} lists no models; pass --api-model"))?
                    .to_owned()
            }
        };
        Ok(Self {
            client,
            base,
            model,
        })
    }
}

/// llama.cpp's `timings`, as teeny-api returns them.
#[derive(Debug, Deserialize)]
struct WireTimings {
    cache_n: u32,
    prompt_n: u32,
    prompt_ms: f64,
    predicted_n: u32,
    predicted_ms: f64,
    #[serde(default)]
    queued_ms: f64,
    #[serde(default)]
    first_token_ms: f64,
    #[serde(default)]
    predicted_latencies_ms: Vec<f64>,
}

fn duration_ms(ms: f64) -> Duration {
    Duration::from_secs_f64(ms.max(0.0) / 1e3)
}

impl From<WireTimings> for Timings {
    fn from(w: WireTimings) -> Self {
        Timings {
            queued: duration_ms(w.queued_ms),
            prompt_tokens: w.prompt_n,
            cached_tokens: w.cache_n,
            prompt: duration_ms(w.prompt_ms),
            first_token: duration_ms(w.first_token_ms),
            generated_tokens: w.predicted_n,
            generation: duration_ms(w.predicted_ms),
            token_latencies: w
                .predicted_latencies_ms
                .into_iter()
                .map(duration_ms)
                .collect(),
        }
    }
}

impl Target for ApiTarget {
    fn describe(&self) -> String {
        format!("{} (teeny-api at {})", self.model, self.base)
    }

    fn chat(&self, text: String, max_tokens: u32) -> BoxFuture<'_, anyhow::Result<Sample>> {
        Box::pin(async move {
            let url = format!("{}/v1/chat/completions", self.base);
            let body = serde_json::json!({
                "model": self.model,
                "messages": [{"role": "user", "content": text}],
                "max_completion_tokens": max_tokens,
            });
            let started = Instant::now();
            let response = self.client.post(&url).json(&body).send().await?;
            let status = response.status();
            let text = response.text().await?;
            let round_trip = started.elapsed();
            ensure!(status.is_success(), "{url} answered {status}: {text}");
            let v: serde_json::Value = serde_json::from_str(&text)?;
            let wire = v.get("timings").cloned().with_context(|| {
                format!("{url} returned no timings (is it teeny-api, with a model that measures itself?)")
            })?;
            let timings = serde_json::from_value::<WireTimings>(wire)?.into();
            Ok(Sample {
                timings,
                round_trip,
            })
        })
    }
}

/// What a test measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Test {
    /// Prompt processing: a prompt of this many tokens, one token generated.
    Prompt(u32),
    /// Text generation: this many tokens generated after a short prompt.
    Generation(u32),
}

impl Test {
    /// Its name, as llama-bench gives it (`pp64`, `tg128`).
    pub fn name(self) -> String {
        match self {
            Test::Prompt(n) => format!("pp{n}"),
            Test::Generation(n) => format!("tg{n}"),
        }
    }
}

/// What to run.
#[derive(Debug, Clone)]
pub struct Workload {
    pub tests: Vec<Test>,
    /// Runs per test that are measured.
    pub repetitions: u32,
    /// Runs per test first, not measured (compilation, caches, clocks settling).
    pub warmup: u32,
}

/// A mean and its sample standard deviation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Summary {
    pub mean: f64,
    pub stddev: f64,
}

impl Summary {
    /// Of `values` (zero for none; no spread for one).
    pub fn of(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let stddev = if values.len() < 2 {
            0.0
        } else {
            (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        };
        Self { mean, stddev }
    }
}

/// Decode step latencies, in milliseconds, over every measured run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Percentiles {
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

/// One measured run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Run {
    pub prompt_n: u32,
    pub cache_n: u32,
    pub prompt_ms: f64,
    pub predicted_n: u32,
    pub predicted_ms: f64,
    pub first_token_ms: f64,
    pub queued_ms: f64,
    pub round_trip_ms: f64,
}

/// What a test measured.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TestResult {
    pub test: String,
    pub kind: Test,
    /// Prompt tokens run, and reused from the cache, per run (mean).
    pub prompt_tokens: f64,
    pub cached_tokens: f64,
    /// Tokens generated per run (mean).
    pub generated_tokens: f64,
    /// The test's rate: prompt tokens per second for `pp`, decode steps per second for `tg`.
    pub tokens_per_second: Summary,
    pub first_token_ms: Summary,
    pub round_trip_ms: Summary,
    /// For `tg`.
    pub latency_ms: Option<Percentiles>,
    pub runs: Vec<Run>,
}

/// A benchmark's results.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub target: String,
    /// Loading the model (in-process), when it was loaded for the benchmark.
    pub load_ms: Option<f64>,
    pub repetitions: u32,
    pub warmup: u32,
    pub results: Vec<TestResult>,
}

/// Prompt words that are a token each.
const FILLER: &str = "the";

/// Hands out the run tags that start every request, from a different place each benchmark
/// so a server's cache keeps nothing from the last one. Fixed width, so every tag is as
/// many tokens as the others.
struct Tags(AtomicU32);

impl Tags {
    fn new() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        Self(AtomicU32::new(seed % 10_000))
    }

    fn next(&self) -> String {
        format!(
            "Run {:04}.",
            self.0.fetch_add(1, Ordering::Relaxed) % 10_000
        )
    }
}

/// A prompt of `words` filler words, opened by a run tag.
fn filler(tag: &str, words: usize) -> String {
    let mut text = tag.to_owned();
    for _ in 0..words {
        text.push(' ');
        text.push_str(FILLER);
    }
    text
}

/// The short prompt of a generation test, opened by a run tag. A model may stop before the
/// test's length (the report says how many tokens it generated); this one keeps
/// FunctionGemma going to the limit whatever the tag.
fn generation_prompt(tag: &str) -> String {
    format!("{tag} List 100 fruits, one per line.")
}

/// How the prompt tokens a request runs grow with its filler words: `fixed + per_word * words`.
#[derive(Debug, Clone, Copy)]
struct Calibration {
    fixed: f64,
    per_word: f64,
}

impl Calibration {
    /// Fits the line from two probes, after a first request that fills the model's cache
    /// with what every request shares (the template's opening), so the probes run only
    /// what their own requests will.
    async fn probe(target: &dyn Target, tags: &Tags) -> anyhow::Result<Self> {
        target.chat(filler(&tags.next(), 1), 1).await?;
        let (short, long) = (8usize, 40usize);
        let a = target
            .chat(filler(&tags.next(), short), 1)
            .await?
            .timings
            .prompt_tokens;
        let b = target
            .chat(filler(&tags.next(), long), 1)
            .await?
            .timings
            .prompt_tokens;
        ensure!(
            b > a,
            "a longer prompt ran no more tokens ({a} then {b}); cannot size prompts"
        );
        let per_word = f64::from(b - a) / (long - short) as f64;
        Ok(Self {
            fixed: f64::from(a) - per_word * short as f64,
            per_word,
        })
    }

    /// Filler words for a prompt that runs `tokens` tokens, corrected against the target
    /// until a request runs that many (or as near as a word allows): a few requests, which
    /// also warm the model up.
    async fn fit(self, target: &dyn Target, tags: &Tags, tokens: u32) -> anyhow::Result<usize> {
        let mut words = self.words(tokens)?;
        for _ in 0..4 {
            let ran = target
                .chat(filler(&tags.next(), words), 1)
                .await?
                .timings
                .prompt_tokens;
            let off = (f64::from(tokens) - f64::from(ran)) / self.per_word;
            if off.abs() < 1.0 {
                break;
            }
            words = (words as f64 + off).round().max(1.0) as usize;
        }
        Ok(words)
    }

    /// Filler words for a prompt that runs `tokens` tokens, by the line.
    fn words(self, tokens: u32) -> anyhow::Result<usize> {
        let words = ((f64::from(tokens) - self.fixed) / self.per_word).round();
        if words < 1.0 {
            bail!(
                "pp{tokens} is shorter than a request's own {:.0} tokens (the chat template and \
                 run tag); ask for a longer prompt",
                self.fixed
            );
        }
        Ok(words as usize)
    }
}

/// Runs `workload` against `target`, telling `progress` what it is doing.
pub async fn run(
    target: &dyn Target,
    workload: &Workload,
    load_ms: Option<f64>,
    mut progress: impl FnMut(&str),
) -> anyhow::Result<Report> {
    ensure!(
        workload.repetitions > 0,
        "there must be at least one repetition"
    );
    let tags = Tags::new();
    progress("calibrating prompt lengths");
    let calibration = Calibration::probe(target, &tags).await?;
    let mut results = Vec::new();
    for &test in &workload.tests {
        let name = test.name();
        let words = match test {
            Test::Prompt(n) => {
                progress(&format!("{name}: sizing the prompt"));
                calibration
                    .fit(target, &tags, n)
                    .await
                    .with_context(|| name.clone())?
            }
            Test::Generation(_) => 0,
        };
        let request = || -> anyhow::Result<(String, u32)> {
            Ok(match test {
                Test::Prompt(_) => (filler(&tags.next(), words), 1),
                Test::Generation(n) => (generation_prompt(&tags.next()), n),
            })
        };
        for i in 0..workload.warmup {
            progress(&format!("{name}: warm-up {}/{}", i + 1, workload.warmup));
            let (text, max) = request()?;
            target
                .chat(text, max)
                .await
                .with_context(|| format!("{name} (warm-up)"))?;
        }
        let mut samples = Vec::new();
        for i in 0..workload.repetitions {
            progress(&format!("{name}: run {}/{}", i + 1, workload.repetitions));
            let (text, max) = request()?;
            samples.push(target.chat(text, max).await.with_context(|| name.clone())?);
        }
        results.push(summarise(test, &samples)?);
    }
    Ok(Report {
        target: target.describe(),
        load_ms,
        repetitions: workload.repetitions,
        warmup: workload.warmup,
        results,
    })
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn summarise(test: Test, samples: &[Sample]) -> anyhow::Result<TestResult> {
    let mean = |f: &dyn Fn(&Timings) -> f64| {
        Summary::of(&samples.iter().map(|s| f(&s.timings)).collect::<Vec<_>>()).mean
    };
    let rates: Vec<f64> = samples
        .iter()
        .map(|s| match test {
            Test::Prompt(_) => s.timings.prompt_tokens_per_second(),
            Test::Generation(_) => s.timings.generation_tokens_per_second(),
        })
        .collect();
    let latency_ms = match test {
        Test::Prompt(_) => None,
        Test::Generation(n) => {
            let all: Vec<Duration> = samples
                .iter()
                .flat_map(|s| s.timings.token_latencies.iter().copied())
                .collect();
            ensure!(
                !all.is_empty(),
                "tg{n}: the model stopped after its first token every time, so there were no \
                 decode steps to measure"
            );
            let p = |q| percentile(&all, q).map_or(0.0, ms);
            Some(Percentiles {
                p50: p(50.0),
                p90: p(90.0),
                p99: p(99.0),
            })
        }
    };
    Ok(TestResult {
        test: test.name(),
        kind: test,
        prompt_tokens: mean(&|t| f64::from(t.prompt_tokens)),
        cached_tokens: mean(&|t| f64::from(t.cached_tokens)),
        generated_tokens: mean(&|t| f64::from(t.generated_tokens)),
        tokens_per_second: Summary::of(&rates),
        first_token_ms: Summary::of(
            &samples
                .iter()
                .map(|s| ms(s.timings.first_token))
                .collect::<Vec<_>>(),
        ),
        round_trip_ms: Summary::of(&samples.iter().map(|s| ms(s.round_trip)).collect::<Vec<_>>()),
        latency_ms,
        runs: samples
            .iter()
            .map(|s| Run {
                prompt_n: s.timings.prompt_tokens,
                cache_n: s.timings.cached_tokens,
                prompt_ms: ms(s.timings.prompt),
                predicted_n: s.timings.generated_tokens,
                predicted_ms: ms(s.timings.generation),
                first_token_ms: ms(s.timings.first_token),
                queued_ms: ms(s.timings.queued),
                round_trip_ms: ms(s.round_trip),
            })
            .collect(),
    })
}

impl Report {
    /// The report as a markdown table, llama-bench style.
    pub fn markdown(&self) -> String {
        let mut out = format!("**{}**", self.target);
        if let Some(load) = self.load_ms {
            out.push_str(&format!(", loaded in {:.1} s", load / 1e3));
        }
        out.push_str(&format!(
            "; {} run(s) per test after {} warm-up\n\n",
            self.repetitions, self.warmup
        ));
        out.push_str(
            "| test | prompt | cached | generated | tokens/s | first token | p50 | p90 | p99 |\n",
        );
        out.push_str(
            "| ---- | -----: | -----: | --------: | -------: | ----------: | --: | --: | --: |\n",
        );
        for r in &self.results {
            let (p50, p90, p99) =
                r.latency_ms
                    .map_or((String::new(), String::new(), String::new()), |l| {
                        (
                            format!("{:.1} ms", l.p50),
                            format!("{:.1} ms", l.p90),
                            format!("{:.1} ms", l.p99),
                        )
                    });
            out.push_str(&format!(
                "| {} | {:.0} | {:.0} | {:.0} | {:.1} ± {:.1} | {:.1} ms | {p50} | {p90} | {p99} |\n",
                r.test,
                r.prompt_tokens,
                r.cached_tokens,
                r.generated_tokens,
                r.tokens_per_second.mean,
                r.tokens_per_second.stddev,
                r.first_token_ms.mean,
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summary_is_mean_and_sample_stddev() {
        let s = Summary::of(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert_eq!(s.mean, 5.0);
        assert!((s.stddev - 2.138).abs() < 1e-3, "{s:?}");
        assert_eq!(
            Summary::of(&[3.0]),
            Summary {
                mean: 3.0,
                stddev: 0.0
            }
        );
        assert_eq!(Summary::of(&[]), Summary::default());
    }

    #[test]
    fn test_tags_are_all_as_long() {
        let tags = Tags(AtomicU32::new(9_998));
        let all: Vec<String> = (0..4).map(|_| tags.next()).collect();
        assert_eq!(all, ["Run 9998.", "Run 9999.", "Run 0000.", "Run 0001."]);
    }

    #[test]
    fn test_calibration_sizes_prompts_and_refuses_short_ones() {
        let c = Calibration {
            fixed: 10.0,
            per_word: 1.0,
        };
        assert_eq!(c.words(64).unwrap(), 54);
        let err = c.words(8).unwrap_err().to_string();
        assert!(err.contains("pp8") && err.contains("10 tokens"), "{err}");
    }

    #[test]
    fn test_wire_timings_come_back_as_durations() {
        let t: Timings = serde_json::from_value::<WireTimings>(serde_json::json!({
            "cache_n": 2, "prompt_n": 64, "prompt_ms": 500.0, "prompt_per_second": 128.0,
            "predicted_n": 3, "predicted_ms": 20.0, "first_token_ms": 510.0,
            "predicted_latencies_ms": [10.0, 10.0]
        }))
        .unwrap()
        .into();
        assert_eq!(
            (t.prompt_tokens, t.cached_tokens, t.generated_tokens),
            (64, 2, 3)
        );
        assert_eq!(t.prompt_tokens_per_second(), 128.0);
        assert_eq!(t.generation_tokens_per_second(), 100.0);
    }
}
