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

//! End-to-end tests of the OpenAI-compatible API against the placeholder `EchoEngine`.

use std::sync::Arc;

use serde_json::{Value, json};
use teeny_llm_core::{EchoEngine, Engine};

async fn spawn_server() -> String {
    let engine: Arc<dyn Engine> = Arc::new(EchoEngine::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, teeny_llm_server::router(engine))
            .await
            .unwrap();
    });
    format!("http://{addr}")
}

async fn post(base: &str, path: &str, body: Value) -> (u16, String) {
    let resp = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap();
    (resp.status().as_u16(), resp.text().await.unwrap())
}

/// The JSON payloads of an SSE body, with the terminating `[DONE]` checked and stripped.
fn sse_payloads(body: &str) -> Vec<Value> {
    let data: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(
        data.last(),
        Some(&"[DONE]"),
        "stream must end with [DONE]: {body}"
    );
    data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

#[tokio::test]
async fn lists_and_retrieves_models() {
    let base = spawn_server().await;
    let list: Value = reqwest::get(format!("{base}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"][0]["id"], EchoEngine::MODEL_ID);
    assert_eq!(list["data"][0]["object"], "model");

    let missing = reqwest::get(format!("{base}/v1/models/gpt-4o"))
        .await
        .unwrap();
    assert_eq!(missing.status().as_u16(), 404);
    let err: Value = missing.json().await.unwrap();
    assert_eq!(err["error"]["code"], "model_not_found");
    assert_eq!(err["error"]["param"], "model");
}

#[tokio::test]
async fn chat_completion() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({
            "model": EchoEngine::MODEL_ID,
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
            ],
            "logit_bias": {"50256": -100}
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "hello world");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(v["choices"][0]["logprobs"].is_null());
    assert_eq!(v["usage"]["completion_tokens"], 2);
    assert_eq!(v["usage"]["total_tokens"], 6);
}

#[tokio::test]
async fn chat_completion_stream_with_usage() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({
            "model": EchoEngine::MODEL_ID,
            "messages": [{"role": "user", "content": "one two three"}],
            "max_completion_tokens": 2,
            "stream": true,
            "stream_options": {"include_usage": true}
        }),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_payloads(&body);
    assert!(
        chunks
            .iter()
            .all(|c| c["object"] == "chat.completion.chunk")
    );
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "one two");

    let n = chunks.len();
    assert_eq!(chunks[n - 2]["choices"][0]["finish_reason"], "length");
    assert_eq!(chunks[n - 1]["choices"], json!([]));
    assert_eq!(chunks[n - 1]["usage"]["completion_tokens"], 2);
    assert!(chunks[..n - 1].iter().all(|c| c.get("usage").is_none()));
}

#[tokio::test]
async fn completion_and_stream() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/completions",
        json!({"model": EchoEngine::MODEL_ID, "prompt": ["a b", "c"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "a b");
    assert_eq!(v["choices"][1]["text"], "c");
    assert_eq!(v["choices"][1]["index"], 1);
    assert_eq!(v["usage"]["completion_tokens"], 3);

    let (status, body) = post(
        &base,
        "/v1/completions",
        json!({"model": EchoEngine::MODEL_ID, "prompt": "x y", "stream": true}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_payloads(&body);
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, "x y");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[tokio::test]
async fn rejects_unsupported_requests_with_openai_errors() {
    let base = spawn_server().await;
    let cases = [
        (
            "/v1/chat/completions",
            json!({"model": EchoEngine::MODEL_ID, "messages": [], "n": 2}),
            400,
            "n",
        ),
        (
            "/v1/chat/completions",
            json!({"model": EchoEngine::MODEL_ID, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]}]}),
            400,
            "messages[0].content[0].type",
        ),
        (
            "/v1/completions",
            json!({"model": EchoEngine::MODEL_ID, "prompt": "x", "echo": true}),
            400,
            "echo",
        ),
        (
            "/v1/chat/completions",
            json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}),
            404,
            "model",
        ),
    ];
    for (path, body, want_status, want_param) in cases {
        let (status, text) = post(&base, path, body).await;
        assert_eq!(status, want_status, "{path}: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["param"], want_param, "{path}: {text}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    let (status, text) = post(&base, "/v1/chat/completions", json!({"model": 1})).await;
    assert_eq!(status, 422, "{text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");

    let (status, text) = post(
        &base,
        "/v1/embeddings",
        json!({"model": EchoEngine::MODEL_ID, "input": "hi"}),
    )
    .await;
    assert_eq!(status, 501, "{text}");
}
