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
//!
//! [`weather`] returns one more field than the model should see: `scene`, which carries what
//! the web demo needs to paint the sky (local time, day or night, wind bearing, cloud cover).
//! Pass the result through [`model_view`] before putting it in a `tool` message, or the model
//! reads those numbers out loud.

use serde_json::{Value, json};

/// How long to wait for Open-Meteo before giving up, in seconds.
const TIMEOUT_SECS: &str = "10";

/// `get_weather`'s declaration, in the OpenAI style the chat template expects.
///
/// The description is doing real work, so do not trim it back to "get the weather". The
/// template puts declarations in the `developer` turn, and for this 270M model that turn is
/// the only place a role sticks: a plain system prompt in the same turn, which the template
/// does support, stopped it answering at all. Here the wording stays in distribution, and
/// naming the ways people actually ask is what gets the tool called for a question like
/// "what is it like in Tokyo?" instead of a paragraph about the city.
///
/// Measured on the board over five phrasings: three of five called the tool with the one-line
/// description, five of five with this one.
///
/// Every word here is load-bearing in a way that is easy to get wrong. An earlier version
/// named two example cities in the `city` description and the model began refusing everywhere
/// else, telling a visitor its tools covered "cities like Tokyo and Lisbon". Say what the tool
/// can do, never what a value might look like.
pub fn declaration() -> Value {
    json!({"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather anywhere in the world: temperature, humidity, \
                        wind and conditions such as rain, snow, fog or clear sky. Use this for \
                        every question that names a place, however it is asked: what it is \
                        like there, how warm or cold it is, whether it is raining or snowing, \
                        or whether someone needs an umbrella or a coat.",
        "parameters": {"type": "object",
            "properties": {"city": {"type": "string",
                // Deliberately no example cities. With "for example Tokyo or Lisbon" here,
                // the model read the examples as the whole list and told a visitor asking
                // about Smolensk that its tools only covered "cities like Tokyo and Lisbon".
                "description": "Any city, town or village in the world, named as the \
                                visitor wrote it"}},
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
                "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,\
                 weather_code,is_day,cloud_cover",
            ),
            ("timezone", "auto"),
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
        // Not for the model; see `model_view`.
        "scene": {
            "weather_code": now["weather_code"],
            "is_day": now["is_day"],
            "cloud_cover": now["cloud_cover"],
            "wind_kmh": now["wind_speed_10m"],
            "wind_from_deg": now["wind_direction_10m"],
            // Local wall-clock at the city, e.g. "2026-11-18T05:00". The sky is painted from
            // its hour, so a 5am answer about Hawaii gets a sunrise and not our own midnight.
            "local_time": now["time"],
            "utc_offset_seconds": w["utc_offset_seconds"],
        },
    }))
}

/// The same result with `scene` removed: what goes back to the model.
///
/// The scene fields exist for the browser. Handing them to a 270M model just invites it to
/// recite a wind bearing and a cloud-cover percentage nobody asked for.
pub fn model_view(result: &Value) -> Value {
    let mut v = result.clone();
    if let Some(obj) = v.as_object_mut() {
        obj.remove("scene");
    }
    v
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

/// Which way the sky is, as far as a sentence can contradict it.
enum Sky {
    /// Something is falling out of it.
    Wet,
    /// Nothing is, and you can see the sun.
    Clear,
    /// Nothing is, but you cannot.
    Dim,
}

/// Does `reply` claim weather that `result` says is not happening?
///
/// The model is 270M parameters and, asked a leading question, will sometimes agree with the
/// premise and then contradict itself from the tool result a clause later: "It is currently
/// raining in Tokyo. The conditions are clear sky." The reading is ground truth and is already
/// on screen, so a sentence that disagrees with it is worse than no sentence, and the caller
/// drops it.
///
/// This only judges whether it rains or shines, since that is the claim that misleads. Being
/// a word or two off about the temperature is not worth suppressing an answer over.
pub fn contradicts(reply: &str, result: &Value) -> bool {
    let sky = match result["conditions"].as_str() {
        Some("drizzle" | "rain" | "rain showers" | "snow" | "snow showers" | "thunderstorm") => {
            Sky::Wet
        }
        Some("clear sky") => Sky::Clear,
        Some("partly cloudy" | "overcast" | "fog") => Sky::Dim,
        // "unknown", or no reading at all: nothing to contradict.
        _ => return false,
    };
    const FALLING: &[&str] = &[
        "rain", "drizzl", "snow", "thunder", "sleet", "hail", "storm",
    ];
    const SHINING: &[&str] = &["clear sky", "sunny", "cloudless"];
    let text = reply.to_lowercase();
    let claims = |words: &[&str]| words.iter().any(|w| claimed(&text, w));
    match sky {
        Sky::Wet => claims(SHINING),
        Sky::Clear => claims(FALLING),
        Sky::Dim => claims(FALLING) || claims(SHINING),
    }
}

/// Is `needle` asserted anywhere in `text`, rather than denied?
///
/// "It is not raining" and "no snow today" agree with a dry reading, so they must not count
/// as claims that it rains.
fn claimed(text: &str, needle: &str) -> bool {
    text.match_indices(needle).any(|(at, _)| !denied(text, at))
}

/// Does a denial sit just before the word at `at`?
fn denied(text: &str, at: usize) -> bool {
    // Enough room for "there is no " and the like, moved up to a character boundary so a
    // degree sign earlier in the sentence cannot panic the slice.
    let mut start = at.saturating_sub(20);
    while start < at && !text.is_char_boundary(start) {
        start += 1;
    }
    let before = &text[start..at];
    ["not ", "no ", "n't ", "without ", "isn", "aren"]
        .iter()
        .any(|d| before.contains(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reading(conditions: &str) -> Value {
        json!({"conditions": conditions, "temperature_c": 15})
    }

    /// Every condition the tool can report, and whether it counts as wet.
    const ALL: &[(&str, bool)] = &[
        ("clear sky", false),
        ("partly cloudy", false),
        ("overcast", false),
        ("fog", false),
        ("drizzle", true),
        ("rain", true),
        ("snow", true),
        ("rain showers", true),
        ("snow showers", true),
        ("thunderstorm", true),
    ];

    #[test]
    fn catches_the_leading_question_contradiction() {
        // Word for word off the board, which is what this guard exists for.
        let reply = "It is currently raining in Tokyo, Japan. The conditions are clear sky, \
                     with a humidity of 46%, and a wind speed of 7.7 km/h.";
        assert!(contradicts(reply, &reading("clear sky")));
    }

    #[test]
    fn every_dry_reading_rejects_every_claim_of_falling_weather() {
        let claims = [
            "It is raining there.",
            "There is heavy rain.",
            "It is drizzling.",
            "It is snowing right now.",
            "Expect thunderstorms.",
            "There is sleet.",
            "A storm is passing through.",
        ];
        for (conditions, wet) in ALL.iter().filter(|(_, wet)| !wet) {
            assert!(!wet);
            for claim in claims {
                assert!(
                    contradicts(claim, &reading(conditions)),
                    "{claim:?} should contradict {conditions:?}"
                );
            }
        }
    }

    #[test]
    fn every_wet_reading_rejects_every_claim_of_sunshine() {
        let claims = [
            "The sky is clear sky.",
            "It is sunny.",
            "A cloudless afternoon.",
        ];
        for (conditions, wet) in ALL.iter().filter(|(_, wet)| *wet) {
            assert!(wet);
            for claim in claims {
                assert!(
                    contradicts(claim, &reading(conditions)),
                    "{claim:?} should contradict {conditions:?}"
                );
            }
        }
    }

    #[test]
    fn a_reading_never_contradicts_its_own_wording() {
        for (conditions, _) in ALL {
            let reply = format!("The current weather is:\n*   Conditions: {conditions}");
            assert!(
                !contradicts(&reply, &reading(conditions)),
                "{conditions:?} contradicted itself"
            );
        }
    }

    #[test]
    fn allows_a_denial() {
        let dry = [
            "It is not raining in Tokyo.",
            "There is no snow today.",
            "It isn't raining.",
            "A day without rain.",
            "There are no thunderstorms.",
        ];
        for reply in dry {
            assert!(!contradicts(reply, &reading("clear sky")), "{reply:?}");
            assert!(!contradicts(reply, &reading("overcast")), "{reply:?}");
        }
    }

    #[test]
    fn cloud_and_fog_reject_claims_in_both_directions() {
        for conditions in ["partly cloudy", "overcast", "fog"] {
            assert!(contradicts("It is snowing.", &reading(conditions)));
            assert!(contradicts("It is sunny.", &reading(conditions)));
        }
    }

    #[test]
    fn says_nothing_about_a_result_it_cannot_read() {
        let unreadable = [
            json!({"error": "weather service unreachable"}),
            json!({"conditions": "unknown"}),
            json!({}),
            json!({"conditions": 7}),
        ];
        for result in unreadable {
            assert!(!contradicts("It is raining.", &result), "{result}");
        }
    }

    #[test]
    fn survives_wording_it_was_not_designed_for() {
        // Must not panic on a multi-byte character sitting in the window before a claim.
        assert!(contradicts(
            "Es regnet. 15 °C and raining.",
            &reading("clear sky")
        ));
        assert!(!contradicts("", &reading("clear sky")));
        assert!(!contradicts("15 °C", &reading("clear sky")));
    }

    #[test]
    fn maps_every_wmo_code_the_api_can_send() {
        let expected: &[(u64, &str)] = &[
            (0, "clear sky"),
            (1, "partly cloudy"),
            (2, "partly cloudy"),
            (3, "overcast"),
            (45, "fog"),
            (48, "fog"),
            (51, "drizzle"),
            (53, "drizzle"),
            (55, "drizzle"),
            (56, "drizzle"),
            (57, "drizzle"),
            (61, "rain"),
            (63, "rain"),
            (65, "rain"),
            (66, "rain"),
            (67, "rain"),
            (71, "snow"),
            (73, "snow"),
            (75, "snow"),
            (77, "snow"),
            (80, "rain showers"),
            (81, "rain showers"),
            (82, "rain showers"),
            (85, "snow showers"),
            (86, "snow showers"),
            (95, "thunderstorm"),
            (96, "thunderstorm"),
            (99, "thunderstorm"),
            (4, "unknown"),
            (44, "unknown"),
            (100, "unknown"),
            (999, "unknown"),
        ];
        for (code, want) in expected {
            assert_eq!(conditions(*code), *want, "code {code}");
        }
    }

    #[test]
    fn the_declaration_says_enough_to_get_itself_called() {
        let d = declaration();
        let f = &d["function"];
        assert_eq!(f["name"], "get_weather");
        assert_eq!(f["parameters"]["required"][0], "city");
        let description = f["description"].as_str().unwrap().to_lowercase();
        // The one-line version left the model guessing; these are the askings it needs to
        // recognise. See the note on `declaration`.
        for cue in ["what it is like", "warm", "raining", "snowing", "umbrella"] {
            assert!(
                description.contains(cue),
                "declaration never mentions {cue:?}"
            );
        }
    }

    #[test]
    fn the_declaration_names_no_example_cities() {
        // Naming two cities in the `city` description taught the model that those were the
        // only ones it had: asked about Smolensk it answered that its tools covered "cities
        // like Tokyo and Lisbon". An example of a value reads as the whole set of values.
        let d = declaration();
        let text = d.to_string().to_lowercase();
        for city in [
            "tokyo", "lisbon", "cairo", "paris", "london", "new york", "berlin",
        ] {
            assert!(
                !text.contains(city),
                "the declaration names {city:?}; the model will treat it as the whole list"
            );
        }
        let city_param = d["function"]["parameters"]["properties"]["city"]["description"]
            .as_str()
            .unwrap()
            .to_lowercase();
        assert!(
            city_param.contains("world") || city_param.contains("any"),
            "the city parameter should say that anywhere counts: {city_param:?}"
        );
    }

    #[test]
    fn model_view_hides_the_scene_and_keeps_the_rest() {
        let full = json!({
            "city": "Tokyo, Japan", "temperature_c": 15, "conditions": "clear sky",
            "humidity_percent": 46, "wind_kmh": 7.7,
            "scene": {"weather_code": 0, "cloud_cover": 5},
        });
        let seen = model_view(&full);
        assert!(
            seen.get("scene").is_none(),
            "the model can still see the scene"
        );
        for kept in [
            "city",
            "temperature_c",
            "conditions",
            "humidity_percent",
            "wind_kmh",
        ] {
            assert_eq!(seen[kept], full[kept], "model_view dropped {kept}");
        }
        // The original is untouched, since the browser still needs it.
        assert!(full.get("scene").is_some());
    }
}
