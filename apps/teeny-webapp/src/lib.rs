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

//! `teeny-webapp`: a browser UI for [`teeny-api`], embedded in one binary.
//!
//! The UI is a Vue app (in `ui/`), built by the build script with npm and embedded as
//! [`ASSETS`]. It has two views:
//!
//! - **Chat**, with a model teeny-api serves: a system prompt, max tokens, temperature, and
//!   OpenAI-style tool declarations, whose calls it shows for you to answer with each result;
//! - **Console**, following teeny-api's `/v1/events`: the models loading and served, and
//!   each API request as it starts and finishes.
//!
//! [`app`] serves the UI and `/config.json`, which tells it where teeny-api is; the browser
//! then talks to teeny-api directly (over HTTP, and a websocket for the events), so this
//! server holds no models and proxies nothing.
//!
//! [`teeny-api`]: https://docs.teenygrad.org/api/teenygrad/teeny_api/

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::State,
    http::{Method, StatusCode, Uri, header},
    response::{IntoResponse, Response},
    routing::get,
};

/// A file of the UI build.
#[derive(Debug)]
pub struct Asset {
    /// Its path in the build, without a leading `/` (`assets/index-1a2b.js`).
    pub path: &'static str,
    pub mime: &'static str,
    pub bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/assets.rs"));

/// The file at `path` in the UI build.
pub fn asset(path: &str) -> Option<&'static Asset> {
    ASSETS.iter().find(|a| a.path == path)
}

/// How the UI is configured.
#[derive(Debug, Clone)]
pub struct Config {
    /// Where teeny-api is (`http://127.0.0.1:8000`), as the browser reaches it.
    pub api_url: String,
}

/// The router serving the UI, configured by `config`: `/config.json`, the build's files, and
/// the app's page for any other path without an extension (its own routes, `/chat`).
pub fn app(config: Config) -> Router {
    Router::new()
        .route("/config.json", get(config_json))
        .fallback(serve)
        .with_state(Arc::new(config))
}

async fn config_json(State(config): State<Arc<Config>>) -> Response {
    (
        [(header::CACHE_CONTROL, "no-cache")],
        Json(serde_json::json!({ "apiUrl": config.api_url })),
    )
        .into_response()
}

async fn serve(method: Method, uri: Uri) -> Response {
    if method != Method::GET && method != Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let path = uri.path().trim_start_matches('/');
    let found = match asset(if path.is_empty() { "index.html" } else { path }) {
        Some(a) => Some(a),
        // The app's own routes (`/chat`) are its page; a missing file is not.
        None if !path.rsplit('/').next().unwrap_or_default().contains('.') => asset("index.html"),
        None => None,
    };
    let Some(asset) = found else {
        return StatusCode::NOT_FOUND.into_response();
    };
    // The build's hashed files never change; the page names the current ones.
    let cache = if asset.path.starts_with("assets/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    (
        [
            (header::CONTENT_TYPE, asset.mime),
            (header::CACHE_CONTROL, cache),
        ],
        asset.bytes,
    )
        .into_response()
}
