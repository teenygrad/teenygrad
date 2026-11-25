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

//! End-to-end tests of `/v1/events`, the websocket of server events.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{BARE, spawn, spawn_with_events};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use teeny_api::{Events, ModelEngine, ModelInfo, ServedModel, ServerEvent};
use tokio_tungstenite::tungstenite::Message;

type Socket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn connect(base: &str) -> Socket {
    let url = format!("{}/v1/events", base.replace("http://", "ws://"));
    tokio_tungstenite::connect_async(url).await.unwrap().0
}

/// The next event, as JSON.
async fn next(socket: &mut Socket) -> Value {
    let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("an event within 5s")
        .expect("the socket open")
        .unwrap();
    match message {
        Message::Text(text) => serde_json::from_str(&text).unwrap(),
        other => panic!("expected a text frame, got {other:?}"),
    }
}

#[tokio::test]
async fn test_events_open_with_hello_listing_the_models() {
    let (base, _events) = spawn_with_events(Vec::new()).await;
    let mut socket = connect(&base).await;
    let hello = next(&mut socket).await;
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(hello["models"][0]["id"], BARE);
    assert!(hello["time"].as_u64().is_some_and(|t| t > 0));
}

#[tokio::test]
async fn test_events_report_each_api_request() {
    let (base, _events) = spawn_with_events(Vec::new()).await;
    let mut socket = connect(&base).await;
    assert_eq!(next(&mut socket).await["type"], "hello");

    // Health checks are not reported; API requests are, started and finished.
    reqwest::get(format!("{base}/health")).await.unwrap();
    let response = reqwest::get(format!("{base}/v1/models/missing"))
        .await
        .unwrap();
    let request_id = response.headers()["x-request-id"]
        .to_str()
        .unwrap()
        .to_owned();
    assert_eq!(response.status().as_u16(), 404);

    let started = next(&mut socket).await;
    assert_eq!(started["type"], "request.started");
    assert_eq!(started["id"], request_id.as_str());
    assert_eq!(
        (&started["method"], &started["path"]),
        (&Value::from("GET"), &Value::from("/v1/models/missing"))
    );
    let finished = next(&mut socket).await;
    assert_eq!(finished["type"], "request.finished");
    assert_eq!(finished["id"], request_id.as_str());
    assert_eq!(finished["status"], 404);
    assert!(finished["duration_ms"].as_u64().is_some());
}

#[tokio::test]
async fn test_events_carry_what_the_server_publishes_to_every_subscriber() {
    let engine = Arc::new(ModelEngine::default());
    let events = Events::new();
    let base = spawn(engine.clone(), events.clone()).await;
    let (mut first, mut second) = (connect(&base).await, connect(&base).await);
    for socket in [&mut first, &mut second] {
        let hello = next(socket).await;
        assert_eq!(hello["models"], Value::Array(Vec::new()));
    }

    // A model added while serving is served, and announced.
    let info = ModelInfo {
        id: "late".into(),
        created: 0,
        owned_by: "teenygrad".into(),
    };
    events.publish(ServerEvent::ModelLoading {
        model: "late".into(),
        detail: "compiling".into(),
    });
    engine.add(ServedModel::new(info.clone())).unwrap();
    events.publish(ServerEvent::ModelAdded { model: info.into() });
    for socket in [&mut first, &mut second] {
        let loading = next(socket).await;
        assert_eq!(
            (&loading["type"], &loading["detail"]),
            (&Value::from("model.loading"), &Value::from("compiling"))
        );
        let added = next(socket).await;
        assert_eq!(added["type"], "model.added");
        assert_eq!(added["model"]["id"], "late");
    }
    let listed: Value = reqwest::get(format!("{base}/v1/models/late"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["id"], "late");

    // A client closing leaves the others subscribed.
    first.close(None).await.unwrap();
    events.publish(ServerEvent::Heartbeat);
    assert_eq!(next(&mut second).await["type"], "request.started");
    assert_eq!(next(&mut second).await["type"], "request.finished");
    assert_eq!(next(&mut second).await["type"], "heartbeat");
    second.send(Message::Text("ignored".into())).await.unwrap();
    events.publish(ServerEvent::Heartbeat);
    assert_eq!(next(&mut second).await["type"], "heartbeat");
}

#[cfg(feature = "llm")]
#[tokio::test]
async fn test_events_report_each_generations_timings() {
    use teeny_llm::chat::EchoChatModel;
    use teeny_llm::models::LoadedModel;

    let echo = ServedModel::from(LoadedModel::from_chat(Arc::new(EchoChatModel)));
    let (base, _events) = spawn_with_events(vec![echo]).await;
    let mut socket = connect(&base).await;
    assert_eq!(next(&mut socket).await["type"], "hello");

    for stream in [false, true] {
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/chat/completions"))
            .json(&serde_json::json!({
                "model": EchoChatModel::MODEL_ID,
                "messages": [{"role": "user", "content": "one two three"}],
                "stream": stream,
            }))
            .send()
            .await
            .unwrap();
        let request_id = response.headers()["x-request-id"]
            .to_str()
            .unwrap()
            .to_owned();
        response.text().await.unwrap();

        let mut seen = Vec::new();
        let stats = loop {
            let event = next(&mut socket).await;
            if event["type"] == "request.stats" {
                break event;
            }
            seen.push(event["type"].clone());
        };
        assert_eq!(stats["id"], request_id.as_str(), "stream {stream}");
        assert_eq!(stats["model"], EchoChatModel::MODEL_ID);
        assert_eq!(stats["timings"]["predicted_n"], 3);
        assert_eq!(stats["timings"]["prompt_n"], 3);
        assert!(seen.contains(&Value::from("request.started")), "{seen:?}");
        // Drain this request's finish (which may come before or after its stats).
        if !seen.contains(&Value::from("request.finished")) {
            assert_eq!(next(&mut socket).await["type"], "request.finished");
        }
    }
}
