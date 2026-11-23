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
    // The model sometimes puts the space inside the string: `<escape> Ibarreta<escape>`. That
    // reaches the geocoder as `%20Ibarreta`, which it happens to forgive, and reads as
    // `" Ibarreta"` on the card. Neither is worth passing on.
    let city = city.trim();
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

/// The place the visitor actually named, when the model garbled it.
///
/// Typing "Novosibirsk" got `get_weather(city: "Novibirsk")`, a syllable short, and the
/// geocoder rightly found nothing. The visitor's own words are the better evidence of what
/// was meant, so when a lookup finds no such place this looks through what they typed for a
/// word or short phrase close enough to be the same name misspelled.
///
/// Deliberately conservative: it only ever returns text the visitor themself wrote, and only
/// when it is a near miss, so a model that invents a city outright is still refused rather
/// than quietly answered about somewhere else.
pub fn nearest_from(message: &str, attempted: &str) -> Option<String> {
    let want = attempted.trim().to_lowercase();
    if want.len() < 4 {
        return None;
    }
    let words: Vec<&str> = message
        .split(|c: char| !c.is_alphanumeric() && c != '-' && c != '\'')
        .filter(|w| !w.is_empty())
        .collect();
    // A name can be two or three words ("New York", "Rio de Janeiro").
    let mut best: Option<(usize, String)> = None;
    for n in 1..=3 {
        for start in 0..words.len().saturating_sub(n - 1) {
            let phrase = words[start..start + n].join(" ");
            let lower = phrase.to_lowercase();
            if lower == want {
                return None; // the model copied it correctly; the place really is unknown
            }
            let d = edit_distance(&lower, &want);
            if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                best = Some((d, phrase));
            }
        }
    }
    // Within a quarter of the name's length, which catches a dropped syllable and little else.
    let allowed = (want.chars().count() / 4).max(1);
    best.filter(|(d, _)| *d <= allowed).map(|(_, p)| p)
}

/// Levenshtein distance, two rows at a time.
fn edit_distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if a.is_empty() || b.is_empty() {
        return a.len().max(b.len());
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut row = vec![0usize; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = usize::from(ca != cb);
            row[j + 1] = (prev[j] + cost).min(prev[j + 1] + 1).min(row[j] + 1);
        }
        std::mem::swap(&mut prev, &mut row);
    }
    prev[b.len()]
}

/// Does `reply` quote a measurement the tool did not return?
///
/// The guard above catches a claim about rain where the reading says clear. This catches the
/// quieter failure: numbers that sound right and are not. Asked about two cities at once the
/// model looks one up and then blends both readings, reporting 16.3 C and 79% where the tool
/// returned 17.7 and 40, with nothing about the sky to give it away.
///
/// Only figures written with a unit the tool itself reports are judged, so a model converting
/// to Fahrenheit or mentioning a year is left alone, and a figure counts as sourced if it
/// matches any of the three readings rather than the one its label claims. The question is
/// whether the number came from the tool at all.
pub fn misquotes(reply: &str, result: &Value) -> bool {
    let known: Vec<f64> = ["temperature_c", "humidity_percent", "wind_kmh"]
        .iter()
        .filter_map(|f| result[*f].as_f64())
        .collect();
    if known.is_empty() {
        return false;
    }
    measurements(reply)
        .iter()
        .any(|q| !known.iter().any(|k| (k - q).abs() <= 0.6))
}

/// Every number in `text` written with a unit this tool reports: `17.7 °C`, `40%`, `8.6 km/h`.
fn measurements(text: &str) -> Vec<f64> {
    let lower = text.to_lowercase();
    let bytes = lower.as_bytes();
    let mut found = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !bytes[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let start = if i > 0 && bytes[i - 1] == b'-' {
            i - 1
        } else {
            i
        };
        let mut end = i;
        while end < bytes.len() && (bytes[end].is_ascii_digit() || bytes[end] == b'.') {
            end += 1;
        }
        let number = lower[start..end].trim_end_matches('.');
        let after = lower[end..].trim_start();
        // `°c` and `c` only when the `c` is not the start of a word like "cloud".
        let celsius = after.starts_with("°c")
            || after.starts_with("degrees c")
            || after.starts_with("degree c")
            || (after.starts_with('c')
                && !after[1..].starts_with(|c: char| c.is_ascii_alphabetic()));
        let unit = celsius
            || after.starts_with('%')
            || after.starts_with("percent")
            || after.starts_with("km/h")
            || after.starts_with("kph")
            || after.starts_with("kilometres per hour")
            || after.starts_with("kilometers per hour");
        if unit && let Ok(v) = number.parse::<f64>() {
            found.push(v);
        }
        i = end;
    }
    found
}

/// The same result with `scene` removed: what goes back to the model.
///
/// The scene fields exist for the browser. Handing them to a 270M model just invites it to
/// recite a wind bearing and a cloud-cover percentage nobody asked for.
pub fn model_view(result: &Value) -> Value {
    let mut v = result.clone();
    if let Some(obj) = v.as_object_mut() {
        obj.remove("scene");
        // The visitor should see that their spelling was used instead; the model telling them
        // about its own misspelling would only confuse matters.
        obj.remove("asked_for");
    }
    v
}

/// A WMO weather code in words. The web demo also keys its icon off this text, so the
/// wording is part of the interface: keep the arms in step with `conditionIcon` in
/// `assets/functiongemma/index.html`.
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
    fn test_catches_the_leading_question_contradiction() {
        // Word for word off the board, which is what this guard exists for.
        let reply = "It is currently raining in Tokyo, Japan. The conditions are clear sky, \
                     with a humidity of 46%, and a wind speed of 7.7 km/h.";
        assert!(contradicts(reply, &reading("clear sky")));
    }

    #[test]
    fn test_every_dry_reading_rejects_every_claim_of_falling_weather() {
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
    fn test_every_wet_reading_rejects_every_claim_of_sunshine() {
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
    fn test_a_reading_never_contradicts_its_own_wording() {
        for (conditions, _) in ALL {
            let reply = format!("The current weather is:\n*   Conditions: {conditions}");
            assert!(
                !contradicts(&reply, &reading(conditions)),
                "{conditions:?} contradicted itself"
            );
        }
    }

    #[test]
    fn test_allows_a_denial() {
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
    fn test_cloud_and_fog_reject_claims_in_both_directions() {
        for conditions in ["partly cloudy", "overcast", "fog"] {
            assert!(contradicts("It is snowing.", &reading(conditions)));
            assert!(contradicts("It is sunny.", &reading(conditions)));
        }
    }

    #[test]
    fn test_says_nothing_about_a_result_it_cannot_read() {
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
    fn test_survives_wording_it_was_not_designed_for() {
        // Must not panic on a multi-byte character sitting in the window before a claim.
        assert!(contradicts(
            "Es regnet. 15 °C and raining.",
            &reading("clear sky")
        ));
        assert!(!contradicts("", &reading("clear sky")));
        assert!(!contradicts("15 °C", &reading("clear sky")));
    }

    #[test]
    fn test_maps_every_wmo_code_the_api_can_send() {
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
            (97, "thunderstorm"),
            (98, "thunderstorm"),
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
    fn test_the_declaration_says_enough_to_get_itself_called() {
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
    fn test_a_hailstorm_is_reported_as_a_thunderstorm() {
        // 96 and 99 are "thunderstorm with slight hail" and "with heavy hail", and Open-Meteo
        // will also sell us hail by the millimetre as its own variable. The demo wants neither:
        // every storm reads as one word and the page draws one storm for it, so folding the
        // hail codes in is the decision, not an oversight. This test is here so that splitting
        // them out later is a deliberate act with a failing test to notice, rather than a
        // quiet change that leaves the page drawing rain at a hailstorm.
        for code in [95, 96, 97, 98, 99] {
            assert_eq!(conditions(code), "thunderstorm", "code {code}");
        }
    }

    #[test]
    fn test_a_storm_claimed_over_a_clear_sky_is_caught_however_it_is_worded() {
        // `conditions` never says "hail" or "sleet", but a model is free to, and the guard has
        // to catch it like any other claim that something is falling.
        for claim in [
            "There is hail in Tokyo.",
            "Expect thunderstorms this afternoon.",
            "Hail and sleet are falling.",
            "A thunderstorm is passing over.",
        ] {
            assert!(contradicts(claim, &reading("clear sky")), "{claim:?}");
        }
        // And must not fire when the reading agrees that a storm is happening.
        for claim in ["There is hail.", "A thunderstorm is overhead."] {
            assert!(!contradicts(claim, &reading("thunderstorm")), "{claim:?}");
        }
    }

    #[test]
    fn test_the_declaration_names_no_example_cities() {
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

    fn full_reading() -> Value {
        json!({"city": "Lisbon, Portugal", "temperature_c": 17.7, "conditions": "overcast",
               "humidity_percent": 40, "wind_kmh": 8.6})
    }

    #[test]
    fn test_catches_numbers_blended_from_another_city() {
        // The real one: "Weather in Tokyo and Lisbon?" looked up one and reported both.
        let reply = "The current weather is 16.3°C with 79% humidity and wind at 10.8 km/h.";
        assert!(misquotes(reply, &full_reading()));
    }

    #[test]
    fn test_allows_the_numbers_the_tool_returned() {
        let replies = [
            "17.7°C, 40% humidity, 8.6 km/h",
            "It is 17.7 degrees C with a humidity of 40 percent.",
            "*   **Temperature:** 17.7°C\n*   **Wind:** 8.6 km/h",
            // Rounded a little, and labels swapped around: still the tool's numbers.
            "About 18 C, wind 8.6 kph.",
        ];
        for reply in replies {
            assert!(!misquotes(reply, &full_reading()), "{reply:?}");
        }
    }

    #[test]
    fn test_ignores_figures_that_are_not_its_units() {
        let replies = [
            "17.7°C, which is 63.9°F.",
            "Checked at 22:45 on 2026-11-19.",
            "Cloud cover is heavy and there are 3 layers.",
            "17.7°C under 100 metres of cloud.",
        ];
        for reply in replies {
            assert!(!misquotes(reply, &full_reading()), "{reply:?}");
        }
    }

    #[test]
    fn test_says_nothing_when_there_is_no_reading_to_check_against() {
        assert!(!misquotes("It is 99°C.", &json!({"error": "unreachable"})));
        assert!(!misquotes("", &full_reading()));
    }

    #[test]
    fn test_recovers_a_city_the_model_misspelled() {
        // Word for word off the board: the visitor typed Novosibirsk, the model dropped a
        // syllable, and the geocoder found nothing.
        assert_eq!(
            nearest_from("Novosibirsk", "Novibirsk").as_deref(),
            Some("Novosibirsk")
        );
        assert_eq!(
            nearest_from("What is the weather in Reykjavik?", "Reykavik").as_deref(),
            Some("Reykjavik")
        );
        assert_eq!(
            nearest_from("weather in Rio de Janeiro please", "Rio de Janero").as_deref(),
            Some("Rio de Janeiro")
        );
    }

    #[test]
    fn test_leaves_a_place_that_simply_does_not_exist() {
        // The model copied the visitor faithfully; the place is the problem, not the spelling,
        // and answering about somewhere else would be worse than saying so.
        assert_eq!(nearest_from("Weather in Zzyzxville?", "Zzyzxville"), None);
        // Nothing in the message is close, so there is nothing to fall back to.
        assert_eq!(nearest_from("Tell me about the sea", "Atlantis"), None);
        // Too short to judge: "Ise" and "Oslo" should not swap places.
        assert_eq!(nearest_from("Oslo", "Ise"), None);
    }

    #[test]
    fn test_never_invents_a_name_the_visitor_did_not_type() {
        for (message, attempted) in [
            ("Weather in Paris?", "Berlin"),
            ("How warm is it in Cairo?", "Tokyo"),
            ("", "Lisbon"),
        ] {
            let got = nearest_from(message, attempted);
            if let Some(found) = got {
                assert!(
                    message.to_lowercase().contains(&found.to_lowercase()),
                    "{found:?} is not in {message:?}"
                );
            }
        }
    }

    #[test]
    fn test_model_view_hides_the_scene_and_keeps_the_rest() {
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
        let corrected = json!({"city": "Novosibirsk, Russia", "asked_for": "Novibirsk"});
        assert!(
            model_view(&corrected).get("asked_for").is_none(),
            "the model can see the correction"
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
