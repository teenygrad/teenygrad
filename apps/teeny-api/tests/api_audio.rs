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

//! End-to-end tests of the audio endpoints (transcriptions, translations, speech),
//! against the reference echo models served by `ModelEngine`.

mod common;

use std::sync::Arc;

use common::{BARE, post, spawn_server as spawn_with};
use serde_json::{Value, json};
use teeny_api::ServedModel;
use teeny_audio::codec::{Pcm, encode_wav};
use teeny_audio::speech::EchoSpeechModel;
use teeny_audio::transcription::EchoTranscriptionModel;

const TRANSCRIBE: &str = EchoTranscriptionModel::MODEL_ID;

/// A server over the audio echo models.
async fn spawn_server() -> String {
    spawn_with(vec![
        ServedModel::from_transcription(Arc::new(EchoTranscriptionModel)),
        ServedModel::from_speech(Arc::new(EchoSpeechModel)),
    ])
    .await
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
        (&[("model", BARE)], "talk.wav", "model"),
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
    let pcm = teeny_audio::codec::decode_wav(&wav).unwrap();
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
            json!({"model": BARE, "input": "hi", "voice": "alloy", "response_format": "wav"}),
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
