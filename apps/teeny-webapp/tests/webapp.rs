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

//! The embedded UI, served.

use teeny_webapp::{ASSETS, Config, app};

async fn spawn() -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let config = Config {
        api_url: "http://api.example:8000".into(),
    };
    tokio::spawn(async move { axum::serve(listener, app(config)).await.unwrap() });
    format!("http://{addr}")
}

async fn get(url: String) -> (u16, String, String) {
    let response = reqwest::get(url).await.unwrap();
    let mime = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    (
        response.status().as_u16(),
        mime,
        response.text().await.unwrap_or_default(),
    )
}

#[tokio::test]
async fn test_serves_the_page_for_the_apps_routes() {
    let base = spawn().await;
    let (status, mime, page) = get(format!("{base}/")).await;
    assert_eq!((status, mime.as_str()), (200, "text/html; charset=utf-8"));
    assert!(page.contains("<html"), "{page}");
    for route in ["/chat", "/console", "/chat/deep/link"] {
        let (status, _, body) = get(format!("{base}{route}")).await;
        assert_eq!((status, body.as_str()), (200, page.as_str()), "{route}");
    }
    let (missing, _, _) = get(format!("{base}/assets/missing.js")).await;
    assert_eq!(missing, 404);
}

#[tokio::test]
async fn test_serves_the_config_and_every_built_file() {
    let base = spawn().await;
    let (status, _, config) = get(format!("{base}/config.json")).await;
    assert_eq!(status, 200);
    let config: serde_json::Value = serde_json::from_str(&config).unwrap();
    assert_eq!(config["apiUrl"], "http://api.example:8000");
    assert!(ASSETS.iter().any(|a| a.path == "index.html"));
    for asset in ASSETS {
        let response = reqwest::get(format!("{base}/{}", asset.path))
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), 200, "{}", asset.path);
        assert_eq!(
            response.headers()["content-type"],
            asset.mime,
            "{}",
            asset.path
        );
        assert_eq!(
            response.bytes().await.unwrap().as_ref(),
            asset.bytes,
            "{}",
            asset.path
        );
    }
}

#[tokio::test]
async fn test_refuses_anything_but_reads() {
    let base = spawn().await;
    let status = reqwest::Client::new()
        .post(format!("{base}/chat"))
        .send()
        .await
        .unwrap()
        .status();
    assert_eq!(status.as_u16(), 405);
}
