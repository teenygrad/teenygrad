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

//! End-to-end tests of the LLM endpoints (models, chat, completions, embeddings,
//! moderations), against the reference echo models served by `ModelEngine`.

mod common;

use std::sync::Arc;

use common::{post, spawn_server as spawn_with, sse_payloads};
use serde_json::{Value, json};
use teeny_api::ServedModel;
use teeny_llm::chat::EchoChatModel;
use teeny_llm::completion::EchoCompletionModel;
use teeny_llm::embedding::EchoEmbeddingModel;
use teeny_llm::models::LoadedModel;
use teeny_llm::moderation::EchoModerationModel;

const ECHO: &str = EchoChatModel::MODEL_ID;

const COMPLETE: &str = EchoCompletionModel::MODEL_ID;

/// A server over the LLM echo models.
async fn spawn_server() -> String {
    let models = [
        LoadedModel::from_chat(Arc::new(EchoChatModel)),
        LoadedModel::from_completion(Arc::new(EchoCompletionModel)),
        LoadedModel::from_embedding(Arc::new(EchoEmbeddingModel)),
        LoadedModel::from_moderation(Arc::new(EchoModerationModel)),
    ];
    spawn_with(models.into_iter().map(ServedModel::from).collect()).await
}

#[tokio::test]
async fn test_lists_and_retrieves_models() {
    let base = spawn_server().await;
    let list: Value = reqwest::get(format!("{base}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(list["object"], "list");
    assert_eq!(list["data"][0]["id"], ECHO);
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
async fn test_chat_completion() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({
            "model": ECHO,
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": [{"type": "text", "text": "hello world"}]}
            ],
            "store": false
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
    // llama.cpp's timings, from the model's own measurements.
    let t = &v["timings"];
    assert_eq!(
        (&t["prompt_n"], &t["cache_n"], &t["predicted_n"]),
        (&json!(4), &json!(0), &json!(2))
    );
    assert_eq!(
        t["predicted_latencies_ms"].as_array().map(Vec::len),
        Some(1)
    );
    for key in [
        "prompt_ms",
        "prompt_per_second",
        "predicted_ms",
        "predicted_per_second",
        "first_token_ms",
    ] {
        assert!(t[key].as_f64().is_some_and(|x| x >= 0.0), "{key}: {t}");
    }
}

#[tokio::test]
async fn test_chat_completion_stream_with_usage() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({
            "model": ECHO,
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
    // The timings come with the usage, and only with it.
    assert_eq!(chunks[n - 1]["timings"]["predicted_n"], 2);
    assert!(chunks[..n - 1].iter().all(|c| c.get("timings").is_none()));

    let (_, body) = post(
        &base,
        "/v1/chat/completions",
        json!({"model": ECHO, "messages": [{"role": "user", "content": "hi"}], "stream": true}),
    )
    .await;
    assert!(
        sse_payloads(&body)
            .iter()
            .all(|c| c.get("timings").is_none() && c["choices"] != json!([])),
        "without include_usage, no chunk without choices: {body}"
    );
}

#[tokio::test]
async fn test_completion_and_stream() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/completions",
        json!({"model": COMPLETE, "prompt": ["a b", "c"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert_eq!(v["choices"][0]["text"], "a b");
    assert_eq!(v["choices"][1]["text"], "c");
    assert_eq!(v["choices"][1]["index"], 1);
    assert_eq!(v["usage"]["completion_tokens"], 3);
    assert_eq!(v["timings"]["prompt_n"], 3);
    assert_eq!(v["timings"]["predicted_n"], 3);

    let (status, body) = post(
        &base,
        "/v1/completions",
        json!({"model": COMPLETE, "prompt": "x y", "stream": true}),
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

    let (_, body) = post(
        &base,
        "/v1/completions",
        json!({"model": COMPLETE, "prompt": "x y", "stream": true,
               "stream_options": {"include_usage": true}}),
    )
    .await;
    let chunks = sse_payloads(&body);
    let last = chunks.last().unwrap();
    assert_eq!(last["usage"]["completion_tokens"], 2);
    assert_eq!(last["timings"]["predicted_n"], 2);
}

#[tokio::test]
async fn test_rejects_unsupported_requests_with_openai_errors() {
    let base = spawn_server().await;
    let user = json!([{"role": "user", "content": "hi"}]);
    let cases = [
        (
            "/v1/chat/completions",
            json!({"model": ECHO, "messages": []}),
            400,
            "messages",
        ),
        (
            "/v1/chat/completions",
            json!({"model": ECHO, "messages": user, "n": 9}),
            400,
            "n",
        ),
        (
            "/v1/chat/completions",
            json!({"model": ECHO, "messages": user, "logit_bias": {"50256": -100}}),
            400,
            "logit_bias",
        ),
        (
            "/v1/chat/completions",
            json!({"model": ECHO, "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]}]}),
            400,
            "messages[0].content[0]",
        ),
        (
            "/v1/chat/completions",
            json!({"model": ECHO, "messages": user, "tool_choice": "sometimes"}),
            400,
            "tool_choice",
        ),
        (
            "/v1/chat/completions",
            json!({"model": "gpt-4o", "messages": user}),
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

    for (body, param) in [
        (
            json!({"model": COMPLETE, "prompt": "x", "suffix": "}"}),
            "suffix",
        ),
        (
            json!({"model": COMPLETE, "prompt": "x", "best_of": 2, "stream": true}),
            "best_of",
        ),
        (json!({"model": ECHO, "prompt": "x"}), "model"),
    ] {
        let (status, text) = post(&base, "/v1/completions", body).await;
        assert_eq!(status, 400, "{text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["param"], param, "{text}");
    }
}

#[tokio::test]
async fn test_completion_with_echo_n_and_logprobs() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/completions",
        json!({"model": COMPLETE, "prompt": [[5, 6]], "n": 2, "echo": true, "logprobs": 1}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"].as_array().unwrap().len(), 2);
    assert_eq!(v["choices"][1]["text"], "5 6 5 6");
    let lp = &v["choices"][0]["logprobs"];
    assert_eq!(lp["tokens"], json!([" 5", " 6"]));
    assert_eq!(lp["text_offset"], json!([3, 5]));
    assert_eq!(lp["top_logprobs"][0], json!({" 5": 0.0}));
}

#[tokio::test]
async fn test_embeddings_in_both_encodings() {
    let base = spawn_server().await;
    let model = EchoEmbeddingModel::MODEL_ID;
    let (status, body) = post(
        &base,
        "/v1/embeddings",
        json!({"model": model, "input": ["red apple", "green pear"], "dimensions": 8}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][1]["index"], 1);
    assert_eq!(v["data"][0]["embedding"].as_array().unwrap().len(), 8);
    assert_eq!(v["usage"]["prompt_tokens"], 4);

    let (status, body) = post(
        &base,
        "/v1/embeddings",
        json!({"model": model, "input": [[1, 2, 3]], "encoding_format": "base64"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    // 16 little-endian f32s are 64 bytes, 88 base64 digits.
    assert_eq!(v["data"][0]["embedding"].as_str().unwrap().len(), 88);

    for (body, status, param) in [
        (
            json!({"model": model, "input": "hi", "dimensions": 99}),
            400,
            "dimensions",
        ),
        (json!({"model": model, "input": ""}), 400, "input[0]"),
        (
            json!({"model": model, "input": "hi", "encoding_format": "hex"}),
            400,
            "encoding_format",
        ),
        (json!({"model": ECHO, "input": "hi"}), 400, "model"),
    ] {
        let (got, text) = post(&base, "/v1/embeddings", body).await;
        assert_eq!(got, status, "{text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["error"]["param"], param, "{text}");
    }
}

#[tokio::test]
async fn test_chat_completion_with_several_choices() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({"model": ECHO, "messages": [{"role": "user", "content": "a b"}], "n": 2}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"].as_array().unwrap().len(), 2);
    assert_eq!(v["choices"][1]["index"], 1);
    assert_eq!(v["choices"][1]["message"]["content"], "a b");
}

fn shout_tool() -> Value {
    json!([{"type": "function", "function": {
        "name": "shout", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}
    }}])
}

#[tokio::test]
async fn test_chat_completion_with_a_tool_call() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({"model": ECHO, "messages": [{"role": "user", "content": "hi"}],
               "tools": shout_tool(), "tool_choice": "required"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert!(choice["message"]["content"].is_null());
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "shout");
    assert_eq!(call["function"]["arguments"], r#"{"text":"hi"}"#);

    // The result goes back as a tool message, by the call's id.
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({"model": ECHO, "tools": shout_tool(), "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "HI"},
            {"role": "user", "content": "thanks"}
        ]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "thanks");
}

#[tokio::test]
async fn test_chat_completion_streams_tool_call_deltas() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/chat/completions",
        json!({"model": ECHO, "messages": [{"role": "user", "content": "hello"}],
               "tools": shout_tool(), "tool_choice": {"type": "function", "function": {"name": "shout"}},
               "stream": true, "stream_options": {"include_usage": true}}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let chunks = sse_payloads(&body);
    let calls: Vec<&Value> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["tool_calls"].get(0))
        .collect();
    assert!(calls.len() >= 2, "the arguments arrive in pieces: {body}");
    assert_eq!(calls[0]["id"], "call_echo_0");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "shout");
    assert!(calls[1].get("id").is_none() && calls[1]["function"].get("name").is_none());
    let arguments: String = calls
        .iter()
        .filter_map(|c| c["function"]["arguments"].as_str())
        .collect();
    assert_eq!(arguments, r#"{"text":"hello"}"#);
    let n = chunks.len();
    assert_eq!(chunks[n - 2]["choices"][0]["finish_reason"], "tool_calls");
    assert!(chunks[n - 1]["usage"]["total_tokens"].as_u64().is_some());
}

#[tokio::test]
async fn test_moderates_each_input() {
    let base = spawn_server().await;
    // No model named: the engine picks the one that classifies.
    let (status, body) = post(
        &base,
        "/v1/moderations",
        json!({"input": ["a fine day", "flag:harassment flag:hate"]}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert!(v["id"].as_str().unwrap().starts_with("modr-"));
    assert_eq!(v["model"], EchoModerationModel::MODEL_ID);
    assert_eq!(v["results"][0]["flagged"], false);
    let r = &v["results"][1];
    assert_eq!(
        (r["flagged"].as_bool(), r["categories"]["hate"].as_bool()),
        (Some(true), Some(true))
    );
    assert_eq!(r["categories"]["violence"], false);
    assert_eq!(r["category_scores"]["harassment"], 1.0);

    let (status, body) = post(
        &base,
        "/v1/moderations",
        json!({"model": EchoModerationModel::MODEL_ID, "input": [{"type": "image_url", "image_url": {"url": "data:,"}}]}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["param"],
        "input"
    );
    let (status, body) = post(
        &base,
        "/v1/moderations",
        json!({"model": ECHO, "input": "hi"}),
    )
    .await;
    assert_eq!(status, 400, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["error"]["code"],
        "model_not_supported"
    );
}
