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

//! End-to-end tests of the OpenAI-compatible API, against the reference echo models (chat,
//! completions, embeddings, transcriptions, speech, moderations) served by `ModelEngine`.

use std::sync::Arc;

use serde_json::{Value, json};
use teeny_llm_core::audio::{Pcm, encode_wav};
use teeny_llm_core::chat::EchoChatModel;
use teeny_llm_core::completion::EchoCompletionModel;
use teeny_llm_core::embedding::EchoEmbeddingModel;
use teeny_llm_core::models::LoadedModel;
use teeny_llm_core::moderation::EchoModerationModel;
use teeny_llm_core::speech::EchoSpeechModel;
use teeny_llm_core::transcription::EchoTranscriptionModel;
use teeny_llm_core::{Engine, ModelEngine};

const ECHO: &str = EchoChatModel::MODEL_ID;

const COMPLETE: &str = EchoCompletionModel::MODEL_ID;

const TRANSCRIBE: &str = EchoTranscriptionModel::MODEL_ID;

/// A server over the echo models.
async fn spawn_server() -> String {
    let models = vec![
        LoadedModel::from_chat(Arc::new(EchoChatModel)),
        LoadedModel::from_completion(Arc::new(EchoCompletionModel)),
        LoadedModel::from_embedding(Arc::new(EchoEmbeddingModel)),
        LoadedModel::from_transcription(Arc::new(EchoTranscriptionModel)),
        LoadedModel::from_speech(Arc::new(EchoSpeechModel)),
        LoadedModel::from_moderation(Arc::new(EchoModerationModel)),
    ];
    spawn(Arc::new(ModelEngine::new(models).unwrap())).await
}

async fn spawn(engine: Arc<dyn Engine>) -> String {
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
async fn chat_completion() {
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
}

#[tokio::test]
async fn chat_completion_stream_with_usage() {
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
}

#[tokio::test]
async fn completion_and_stream() {
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
}

#[tokio::test]
async fn rejects_unsupported_requests_with_openai_errors() {
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
async fn completion_with_echo_n_and_logprobs() {
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
async fn embeddings_in_both_encodings() {
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
async fn chat_completion_with_several_choices() {
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
async fn chat_completion_with_a_tool_call() {
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
async fn chat_completion_streams_tool_call_deltas() {
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

/// A multipart form: text fields, then `file` as `file_name` holding `audio`.
fn form(fields: &[(&str, &str)], file_name: &str, audio: &[u8]) -> (String, Vec<u8>) {
    let boundary = "teeny-boundary-7f3a";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n"
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n\
             Content-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(audio);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

/// Two seconds of stereo 8 kHz silence, as a WAV file.
fn two_seconds() -> Vec<u8> {
    encode_wav(&Pcm {
        sample_rate: 8000,
        channels: 2,
        samples: vec![0.0; 32_000],
    })
}

async fn post_form(
    base: &str,
    path: &str,
    fields: &[(&str, &str)],
    file_name: &str,
) -> (u16, String, String) {
    let (content_type, body) = form(fields, file_name, &two_seconds());
    let resp = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .unwrap();
    let mime = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default();
    (resp.status().as_u16(), mime, resp.text().await.unwrap())
}

#[tokio::test]
async fn test_transcribes_in_each_response_format() {
    let base = spawn_server().await;
    let path = "/v1/audio/transcriptions";
    let (status, _, body) = post_form(&base, path, &[("model", TRANSCRIBE)], "talk.wav").await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["text"], "2.0 seconds of audio");
    assert_eq!(v["usage"], json!({"type": "duration", "seconds": 2.0}));

    let (_, _, body) = post_form(
        &base,
        path,
        &[
            ("model", TRANSCRIBE),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ],
        "talk.wav",
    )
    .await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        (v["task"].as_str(), v["duration"].as_f64()),
        (Some("transcribe"), Some(2.0))
    );
    assert_eq!(v["words"].as_array().unwrap().len(), 4);
    assert_eq!(v["segments"][0]["end"], 2.0);

    let (_, mime, body) = post_form(
        &base,
        path,
        &[("model", TRANSCRIBE), ("response_format", "srt")],
        "talk.wav",
    )
    .await;
    assert_eq!(mime, "application/x-subrip");
    assert_eq!(
        body,
        "1\n00:00:00,000 --> 00:00:02,000\n2.0 seconds of audio\n\n"
    );

    let (_, mime, body) = post_form(
        &base,
        path,
        &[("model", TRANSCRIBE), ("response_format", "text")],
        "talk.wav",
    )
    .await;
    assert!(mime.starts_with("text/plain"));
    assert_eq!(body, "2.0 seconds of audio\n");
}

#[tokio::test]
async fn test_translates_and_streams_transcripts() {
    let base = spawn_server().await;
    let (status, _, body) = post_form(
        &base,
        "/v1/audio/translations",
        &[("model", TRANSCRIBE)],
        "talk.wav",
    )
    .await;
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(&body).unwrap()["text"],
        "2.0 seconds of audio in English"
    );

    let (_, mime, body) = post_form(
        &base,
        "/v1/audio/transcriptions",
        &[("model", TRANSCRIBE), ("stream", "true")],
        "talk.wav",
    )
    .await;
    assert!(mime.starts_with("text/event-stream"));
    let events: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let deltas: String = events
        .iter()
        .filter(|e| e["type"] == "transcript.text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, "2.0 seconds of audio");
    let done = events.last().unwrap();
    assert_eq!(
        (done["type"].as_str(), done["text"].as_str()),
        (Some("transcript.text.done"), Some("2.0 seconds of audio"))
    );
}

#[tokio::test]
async fn test_transcription_errors_name_the_parameter() {
    let base = spawn_server().await;
    let path = "/v1/audio/transcriptions";
    // Each refusal is a 400 naming the parameter at fault.
    type Case<'a> = (&'a [(&'a str, &'a str)], &'a str, &'a str);
    let cases: [Case; 4] = [
        (&[("model", TRANSCRIBE)], "talk.mp3", "file"),
        (&[("model", TRANSCRIBE)], "talk.txt", "file"),
        (&[("model", ECHO)], "talk.wav", "model"),
        (
            &[("model", TRANSCRIBE), ("timestamp_granularities[]", "word")],
            "talk.wav",
            "timestamp_granularities",
        ),
    ];
    for (fields, file_name, param) in cases {
        let (status, _, body) = post_form(&base, path, fields, file_name).await;
        assert_eq!(status, 400, "{body}");
        let err: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(err["error"]["param"], param, "{body}");
    }
}

#[tokio::test]
async fn test_speaks_as_a_file_or_as_events() {
    let base = spawn_server().await;
    let speak = EchoSpeechModel::MODEL_ID;
    let resp = reqwest::Client::new()
        .post(format!("{base}/v1/audio/speech"))
        .json(&json!({"model": speak, "input": "three short words", "voice": "echo", "response_format": "wav"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    assert_eq!(resp.headers()["content-type"], "audio/wav");
    let wav = resp.bytes().await.unwrap();
    let pcm = teeny_llm_core::audio::decode_wav(&wav).unwrap();
    // Three words of 0.2 s at 24 kHz.
    assert_eq!((pcm.sample_rate, pcm.samples.len()), (24_000, 3 * 4800));

    let (status, body) = post(
        &base,
        "/v1/audio/speech",
        json!({"model": speak, "input": "two words", "voice": "alloy", "response_format": "pcm", "stream_format": "sse"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let events: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0]["type"], "speech.audio.delta");
    assert_eq!(
        events[2],
        json!({"type": "speech.audio.done", "usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}})
    );
}

#[tokio::test]
async fn test_speech_errors_name_the_parameter() {
    let base = spawn_server().await;
    let speak = EchoSpeechModel::MODEL_ID;
    let cases = [
        (
            json!({"model": speak, "input": "hi", "voice": "alloy"}),
            "response_format",
        ),
        (
            json!({"model": speak, "input": "hi", "voice": "onyx", "response_format": "wav"}),
            "voice",
        ),
        (
            json!({"model": speak, "input": "hi", "voice": "alloy", "response_format": "wav", "speed": 5}),
            "speed",
        ),
        (
            json!({"model": ECHO, "input": "hi", "voice": "alloy", "response_format": "wav"}),
            "model",
        ),
    ];
    for (body, param) in cases {
        let (status, text) = post(&base, "/v1/audio/speech", body).await;
        assert_eq!(status, 400, "{text}");
        let err: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(err["error"]["param"], param, "{text}");
    }
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
