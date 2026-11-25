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

//! End-to-end tests of the image endpoints (generations, edits, variations), against the
//! reference echo model served by `ModelEngine`.

mod common;

use std::sync::Arc;

use common::{BARE, post, spawn_server as spawn_with};
use serde_json::{Value, json};
use teeny_api::ServedModel;
use teeny_llm::images::{self, EchoImageModel, ImageFormat, Rgba};

/// A server over the image echo model.
async fn spawn_server() -> String {
    spawn_with(vec![ServedModel::from_images(Arc::new(EchoImageModel))]).await
}

const PAINT: &str = EchoImageModel::MODEL_ID;

/// Decodes standard, padded base64.
fn unbase64(text: &str) -> Vec<u8> {
    let digit = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    };
    let digits: Vec<u8> = text.bytes().filter(|&c| c != b'=').map(digit).collect();
    digits
        .chunks(4)
        .flat_map(|q| {
            let n = q
                .iter()
                .enumerate()
                .fold(0u32, |n, (i, &d)| n | u32::from(d) << (18 - 6 * i));
            n.to_be_bytes()[1..q.len()].to_vec()
        })
        .collect()
}

/// The image in a `b64_json` field.
fn image_of(b64: &Value) -> Rgba {
    images::decode(&unbase64(b64.as_str().unwrap()), "image").unwrap()
}

/// A multipart form of `(name, file name, bytes)` parts.
fn multipart(parts: &[(&str, Option<&str>, Vec<u8>)]) -> (String, Vec<u8>) {
    let boundary = "teeny-boundary-image";
    let mut body = Vec::new();
    for (name, file_name, data) in parts {
        let file = file_name.map_or(String::new(), |f| format!("; filename=\"{f}\""));
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"{file}\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(data);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn post_multipart(
    base: &str,
    path: &str,
    parts: &[(&str, Option<&str>, Vec<u8>)],
) -> (u16, String) {
    let (content_type, body) = multipart(parts);
    let resp = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .header("content-type", content_type)
        .body(body)
        .send()
        .await
        .unwrap();
    (resp.status().as_u16(), resp.text().await.unwrap())
}

fn png(image: &Rgba) -> Vec<u8> {
    images::encode(image, ImageFormat::Png).unwrap()
}

#[tokio::test]
async fn test_generates_images_whole_and_streamed() {
    let base = spawn_server().await;
    let (status, body) = post(
        &base,
        "/v1/images/generations",
        json!({"model": PAINT, "prompt": "a red barn", "n": 2, "size": "256x256", "quality": "low"}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"].as_array().unwrap().len(), 2);
    assert_eq!(
        (
            v["output_format"].as_str(),
            v["quality"].as_str(),
            v["size"].as_str()
        ),
        (Some("png"), Some("low"), Some("256x256"))
    );
    assert_eq!(v["usage"]["input_tokens_details"]["text_tokens"], 3);
    let (first, second) = (
        image_of(&v["data"][0]["b64_json"]),
        image_of(&v["data"][1]["b64_json"]),
    );
    assert_eq!((first.width, first.height), (256, 256));
    assert_ne!(first.get(0, 0), second.get(0, 0), "the n images differ");

    let (status, body) = post(
        &base,
        "/v1/images/generations",
        json!({"model": PAINT, "prompt": "a red barn", "size": "256x256", "stream": true, "partial_images": 2}),
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let events: Vec<Value> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(
        kinds,
        [
            "image_generation.partial_image",
            "image_generation.partial_image",
            "image_generation.completed"
        ]
    );
    assert_eq!(events[1]["partial_image_index"], 1);
    assert_eq!(events[2]["usage"]["output_tokens"], 1);
    assert_eq!(image_of(&events[2]["b64_json"]), first);
}

#[tokio::test]
async fn test_edits_within_the_mask_and_varies() {
    let base = spawn_server().await;
    let grey = Rgba::filled(4, 4, [100, 100, 100, 255]);
    let mut mask = Rgba::filled(4, 4, [0, 0, 0, 255]);
    mask.pixels[3] = 0; // only the top-left pixel is open
    let (status, body) = post_multipart(
        &base,
        "/v1/images/edits",
        &[
            ("model", None, PAINT.into()),
            ("prompt", None, b"paint it".to_vec()),
            ("image[]", Some("grey.png"), png(&grey)),
            ("mask", Some("mask.png"), png(&mask)),
        ],
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let edited = image_of(&v["data"][0]["b64_json"]);
    assert_ne!(edited.get(0, 0), grey.get(0, 0));
    assert_eq!(edited.get(1, 0), grey.get(1, 0));

    let (status, body) = post_multipart(
        &base,
        "/v1/images/variations",
        &[
            ("model", None, PAINT.into()),
            ("image", Some("grey.png"), png(&grey)),
        ],
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let varied = image_of(&serde_json::from_str::<Value>(&body).unwrap()["data"][0]["b64_json"]);
    assert_eq!(varied.get(2, 2), [155, 155, 155, 255]);
}

#[tokio::test]
async fn test_image_errors_name_the_parameter() {
    let base = spawn_server().await;
    let generate = [
        (
            json!({"model": PAINT, "prompt": "x", "response_format": "url"}),
            "response_format",
        ),
        (
            json!({"model": PAINT, "prompt": "x", "output_format": "jpeg"}),
            "output_format",
        ),
        (
            json!({"model": PAINT, "prompt": "x", "size": "300x300"}),
            "size",
        ),
        (
            json!({"model": PAINT, "prompt": "x", "partial_images": 2}),
            "partial_images",
        ),
        (json!({"model": PAINT}), "prompt"),
        (json!({"model": BARE, "prompt": "x"}), "model"),
    ];
    for (body, param) in generate {
        let (status, text) = post(&base, "/v1/images/generations", body).await;
        assert_eq!(status, 400, "{text}");
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap()["error"]["param"],
            param,
            "{text}"
        );
    }
    let small_mask = png(&Rgba::filled(1, 1, [0; 4]));
    let (status, text) = post_multipart(
        &base,
        "/v1/images/edits",
        &[
            ("model", None, PAINT.into()),
            ("prompt", None, b"p".to_vec()),
            ("image", Some("a.png"), png(&Rgba::filled(2, 2, [0; 4]))),
            ("mask", Some("m.png"), small_mask),
        ],
    )
    .await;
    assert_eq!(status, 400, "{text}");
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["error"]["param"],
        "mask"
    );
}
