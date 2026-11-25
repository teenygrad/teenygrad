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

//! [`Events`]: what the server tells `/v1/events` subscribers, as it happens.
//!
//! The server [`publish`](Events::publish)es [`ServerEvent`]s (models loading and added,
//! requests started and finished); each websocket subscriber gets them as JSON text
//! frames, one [`Envelope`] each, after a `hello` with the models served so far. Events
//! are not stored: a subscriber sees those published while it is connected, and one that
//! falls too far behind is told how many it `lagged` by.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use tokio::sync::broadcast;

use crate::openai::models::Model;

/// How many events a slow subscriber may fall behind before it misses some.
const BACKLOG: usize = 1024;

/// One event as sent: what happened (its `type` and fields) and when (`time`, in Unix
/// milliseconds).
///
/// ```json
/// {"type": "model.loading", "time": 1764000000000, "model": "functiongemma3:270m", "detail": "compiling the 1-token step"}
/// ```
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Envelope {
    pub time: u64,
    #[serde(flatten)]
    pub event: ServerEvent,
}

impl Envelope {
    /// `event`, stamped with the time now.
    pub fn now(event: ServerEvent) -> Self {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX));
        Self { time, event }
    }
}

/// Something that happened in the server, by `type`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    /// Sent first, on connecting: the server's version and the models it serves so far.
    #[serde(rename = "hello")]
    Hello { version: String, models: Vec<Model> },

    /// Sent every so often while nothing else is, so a client can tell the connection
    /// is alive.
    #[serde(rename = "heartbeat")]
    Heartbeat,

    /// A model is loading; `detail` says what it is doing (loading can take minutes on a
    /// board).
    #[serde(rename = "model.loading")]
    ModelLoading { model: String, detail: String },

    /// A model has loaded and is served from now on.
    #[serde(rename = "model.added")]
    ModelAdded { model: Model },

    /// A model failed to load.
    #[serde(rename = "model.failed")]
    ModelFailed { model: String, error: String },

    /// An API request arrived. `id` is also its response's `x-request-id`.
    #[serde(rename = "request.started")]
    RequestStarted {
        id: String,
        method: String,
        path: String,
    },

    /// An API request was answered, with `status`, `duration_ms` after it arrived. For
    /// a streamed response that is when the stream started.
    #[serde(rename = "request.finished")]
    RequestFinished {
        id: String,
        method: String,
        path: String,
        status: u16,
        duration_ms: u64,
    },

    /// This subscriber fell behind and missed `missed` events.
    #[serde(rename = "lagged")]
    Lagged { missed: u64 },
}

/// Where the server publishes its [`ServerEvent`]s, for every `/v1/events` subscriber.
/// Cloning it gives another handle on the same stream.
#[derive(Debug, Clone)]
pub struct Events {
    sender: broadcast::Sender<Envelope>,
}

impl Default for Events {
    fn default() -> Self {
        Self::new()
    }
}

impl Events {
    pub fn new() -> Self {
        Self {
            sender: broadcast::channel(BACKLOG).0,
        }
    }

    /// Tells every subscriber connected now about `event`. With none, it is dropped.
    pub fn publish(&self, event: ServerEvent) {
        // Sending fails only when nobody is subscribed, which is fine.
        let _ = self.sender.send(Envelope::now(event));
    }

    /// The events published from now on.
    pub fn subscribe(&self) -> broadcast::Receiver<Envelope> {
        self.sender.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_events_serialize_with_their_type_and_time() {
        let e = Envelope {
            time: 7,
            event: ServerEvent::ModelLoading {
                model: "m".into(),
                detail: "compiling".into(),
            },
        };
        assert_eq!(
            serde_json::to_value(&e).unwrap(),
            serde_json::json!({"type": "model.loading", "time": 7, "model": "m", "detail": "compiling"})
        );
        let beat = Envelope::now(ServerEvent::Heartbeat);
        assert_eq!(serde_json::to_value(&beat).unwrap()["type"], "heartbeat");
        assert!(beat.time > 0);
    }

    #[tokio::test]
    async fn test_subscribers_get_what_is_published_after_they_subscribe() {
        let events = Events::new();
        events.publish(ServerEvent::Heartbeat);
        let mut rx = events.subscribe();
        events.publish(ServerEvent::Lagged { missed: 1 });
        assert_eq!(
            rx.recv().await.unwrap().event,
            ServerEvent::Lagged { missed: 1 }
        );
        assert!(rx.try_recv().is_err());
    }
}
