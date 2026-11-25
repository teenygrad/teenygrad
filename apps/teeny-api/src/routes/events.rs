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

//! `/v1/events`, the websocket of [`ServerEvent`]s, and the middleware that publishes
//! each API request's start and finish.

use std::time::{Duration, Instant};

use axum::{
    extract::{
        Request, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use tokio::sync::broadcast::error::RecvError;

use super::{ServerState, new_id};
use crate::events::{Envelope, Events, ServerEvent};
use crate::openai::models::Model;

/// How long a quiet connection waits before a heartbeat.
const HEARTBEAT: Duration = Duration::from_secs(15);

/// The paths whose requests are not published: the events themselves, and health checks.
const UNTRACKED: [&str; 2] = ["/v1/events", "/health"];

pub(super) async fn events_socket(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
) -> Response {
    ws.on_upgrade(move |socket| serve(socket, state))
}

async fn send(socket: &mut WebSocket, event: &Envelope) -> Result<(), axum::Error> {
    // Serializing these plain structs can't fail; fall back to an empty object rather than panic.
    let text = serde_json::to_string(event).unwrap_or_else(|_| "{}".to_owned());
    socket.send(Message::Text(text.into())).await
}

/// Sends `hello`, then every event published, until the client goes.
async fn serve(mut socket: WebSocket, state: ServerState) {
    // Subscribed before `hello`, so nothing published in between is missed.
    let mut events = state.events.subscribe();
    let hello = ServerEvent::Hello {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        models: state.engine.models().into_iter().map(Model::from).collect(),
    };
    if send(&mut socket, &Envelope::now(hello)).await.is_err() {
        return;
    }
    let mut heartbeat =
        tokio::time::interval_at(tokio::time::Instant::now() + HEARTBEAT, HEARTBEAT);
    loop {
        let event = tokio::select! {
            received = events.recv() => match received {
                Ok(event) => event,
                Err(RecvError::Lagged(missed)) => Envelope::now(ServerEvent::Lagged { missed }),
                Err(RecvError::Closed) => break,
            },
            _ = heartbeat.tick() => Envelope::now(ServerEvent::Heartbeat),
            // The client sends nothing we act on; it closing (or vanishing) ends the stream.
            message = socket.recv() => match message {
                None | Some(Err(_) | Ok(Message::Close(_))) => break,
                Some(Ok(_)) => continue,
            },
        };
        if send(&mut socket, &event).await.is_err() {
            break;
        }
        heartbeat.reset();
    }
}

/// Publishes each API request's start and finish, and gives its response the request's
/// id as `x-request-id`.
pub(super) async fn track(State(events): State<Events>, request: Request, next: Next) -> Response {
    let path = request.uri().path().to_owned();
    if UNTRACKED.contains(&path.as_str()) {
        return next.run(request).await;
    }
    let method = request.method().to_string();
    let id = new_id("req");
    let started = Instant::now();
    events.publish(ServerEvent::RequestStarted {
        id: id.clone(),
        method: method.clone(),
        path: path.clone(),
    });
    let mut response = next.run(request).await;
    events.publish(ServerEvent::RequestFinished {
        id: id.clone(),
        method,
        path,
        status: response.status().as_u16(),
        duration_ms: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    });
    if let Ok(value) = HeaderValue::from_str(&id) {
        response.headers_mut().insert("x-request-id", value);
    }
    response
}
