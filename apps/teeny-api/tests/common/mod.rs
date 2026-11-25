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

//! What every API test shares: a server over some models, and requests to it.
#![allow(dead_code)]

use std::sync::Arc;

use serde_json::Value;
use teeny_api::{Engine, ModelEngine, ModelInfo, ServedModel};

/// A model that serves no API, for requests a model cannot answer.
pub const BARE: &str = "teeny-bare";

/// A server over `models`, and [`BARE`].
pub async fn spawn_server(mut models: Vec<ServedModel>) -> String {
    models.push(ServedModel::new(ModelInfo {
        id: BARE.into(),
        created: 0,
        owned_by: "teenygrad".into(),
    }));
    spawn(Arc::new(ModelEngine::new(models).unwrap())).await
}

pub async fn spawn(engine: Arc<dyn Engine>) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, teeny_api::router(engine))
            .await
            .unwrap();
    });
    format!("http://{addr}")
}

pub async fn post(base: &str, path: &str, body: Value) -> (u16, String) {
    let resp = reqwest::Client::new()
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .unwrap();
    (resp.status().as_u16(), resp.text().await.unwrap())
}

/// The JSON payloads of an SSE body, with the terminating `[DONE]` checked and stripped.
pub fn sse_payloads(body: &str) -> Vec<Value> {
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
