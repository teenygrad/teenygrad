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

//! The weather tool the chat and the web demo both offer FunctionGemma.
//!
//! It is the one demo tool that reads from the outside world and writes nothing, so it is
//! the only one the public demo exposes. [`declaration`] is what the model is told about
//! it; [`weather`] is what actually runs when the model calls it.

use serde_json::{Value, json};

/// How long to wait for Open-Meteo before giving up, in seconds.
const TIMEOUT_SECS: &str = "10";

/// `get_weather`'s declaration, in the OpenAI style the chat template expects.
pub fn declaration() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"]}}})
}

/// Current weather for `city` from Open-Meteo: geocoding, then the forecast API.
///
/// Shells out to `curl`, so the board needs it on `PATH` and a route to the internet; a
/// failure comes back as an error rather than a panic, and the caller hands the model
/// `{"error": ...}` so it can say so in words.
pub fn weather(city: &str) -> anyhow::Result<Value> {
    let get = |url: &str, query: &[(&str, &str)]| -> anyhow::Result<Value> {
        let mut cmd = std::process::Command::new("curl");
        cmd.args(["-sfG", "-m", TIMEOUT_SECS, url]);
        for (k, v) in query {
            cmd.arg("--data-urlencode").arg(format!("{k}={v}"));
        }
        let out = cmd.output()?;
        anyhow::ensure!(out.status.success(), "weather service unreachable");
        Ok(serde_json::from_slice(&out.stdout)?)
    };
    let places = get(
        "https://geocoding-api.open-meteo.com/v1/search",
        &[("name", city), ("count", "1")],
    )?;
    let place = places["results"]
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("no place called {city}"))?;
    let (lat, lon) = (
        place["latitude"].to_string(),
        place["longitude"].to_string(),
    );
    let w = get(
        "https://api.open-meteo.com/v1/forecast",
        &[
            ("latitude", &lat),
            ("longitude", &lon),
            (
                "current",
                "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            ),
        ],
    )?;
    let now = &w["current"];
    Ok(json!({
        "city": format!("{}, {}", place["name"].as_str().unwrap_or(city),
                        place["country"].as_str().unwrap_or("")),
        "temperature_c": now["temperature_2m"],
        "conditions": conditions(now["weather_code"].as_u64().unwrap_or(999)),
        "humidity_percent": now["relative_humidity_2m"],
        "wind_kmh": now["wind_speed_10m"],
    }))
}

/// A WMO weather code in words. The web demo also keys its icon off this text, so the
/// wording is part of the interface: keep the arms in step with `conditionIcon` in
/// `assets/index.html`.
pub fn conditions(code: u64) -> &'static str {
    match code {
        0 => "clear sky",
        1 | 2 => "partly cloudy",
        3 => "overcast",
        45 | 48 => "fog",
        51..=57 => "drizzle",
        61..=67 => "rain",
        71..=77 => "snow",
        80..=82 => "rain showers",
        85 | 86 => "snow showers",
        95..=99 => "thunderstorm",
        _ => "unknown",
    }
}
