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

//! Benchmarks of the echo chat model, in-process and through teeny-api.

use std::sync::Arc;

use teeny_bench::{ApiTarget, LocalTarget, Test, Workload};
use teeny_llm::chat::EchoChatModel;
use teeny_llm::models::LoadedModel;

fn workload() -> Workload {
    Workload {
        tests: vec![Test::Prompt(32), Test::Generation(8)],
        repetitions: 3,
        warmup: 1,
    }
}

#[tokio::test]
async fn test_benchmarks_a_model_in_process() {
    let target = LocalTarget::new(Arc::new(EchoChatModel), "in-process");
    let mut seen = Vec::new();
    let report = teeny_bench::run(&target, &workload(), None, |s| seen.push(s.to_owned()))
        .await
        .unwrap();
    assert_eq!(report.target, "teeny-echo (in-process)");
    assert_eq!(
        seen.first().map(String::as_str),
        Some("calibrating prompt lengths")
    );
    assert!(seen.contains(&"tg8: run 3/3".to_owned()), "{seen:?}");

    let [pp, tg] = &report.results[..] else {
        panic!("{report:?}")
    };
    // The echo model counts words as tokens, so prompts come out the length asked for.
    assert_eq!(
        (pp.test.as_str(), pp.prompt_tokens, pp.runs.len()),
        ("pp32", 32.0, 3)
    );
    assert!(pp.latency_ms.is_none());
    assert_eq!(tg.test, "tg8");
    // It echoes the prompt back, as far as the limit.
    assert_eq!(tg.generated_tokens, 8.0);
    assert!(tg.latency_ms.is_some());

    let table = report.markdown();
    assert!(
        table.contains("| pp32 | 32 |") && table.contains("| tg8 |"),
        "{table}"
    );
    let json = serde_json::to_value(&report).unwrap();
    assert_eq!(
        json["results"][1]["kind"],
        serde_json::json!({"generation": 8})
    );
}

#[tokio::test]
async fn test_benchmarks_a_model_through_teeny_api() {
    let echo = LoadedModel::from_chat(Arc::new(EchoChatModel));
    let engine = Arc::new(teeny_api::ModelEngine::new([echo]).unwrap());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = teeny_api::router(engine, teeny_api::Events::new());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let target = ApiTarget::connect(&base, None).await.unwrap();
    let report = teeny_bench::run(&target, &workload(), None, |_| {})
        .await
        .unwrap();
    assert!(
        report
            .target
            .starts_with("teeny-echo (teeny-api at http://127.0.0.1:"),
        "{}",
        report.target
    );
    assert_eq!(report.results[0].prompt_tokens, 32.0);
    assert_eq!(report.results[1].generated_tokens, 8.0);
    assert!(report.results.iter().all(|r| r.round_trip_ms.mean > 0.0));

    let missing = ApiTarget::connect(&base, Some("no-such-model"))
        .await
        .unwrap();
    let err = teeny_bench::run(&missing, &workload(), None, |_| {})
        .await
        .unwrap_err();
    assert!(format!("{err:#}").contains("404"), "{err:#}");
}
