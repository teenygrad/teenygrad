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

//! Standard base64 (RFC 4648), for the audio and file data the chat API carries inline,
//! with the `base64` crate. Encoding pads; decoding is lenient about what clients send:
//! whitespace (line-wrapped data) is ignored and padding is optional.

use base64::Engine as _;
use base64::alphabet::STANDARD;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};

/// Standard alphabet; pads when encoding, takes either when decoding.
const ENGINE: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// Encodes `bytes`.
pub fn encode(bytes: &[u8]) -> String {
    ENGINE.encode(bytes)
}

/// Decodes `text`, ignoring whitespace; padding is optional. (Only chat takes base64 in.)
#[cfg_attr(not(feature = "llm"), allow(dead_code))]
pub fn decode(text: &str) -> Result<Vec<u8>, String> {
    let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    ENGINE
        .decode(&digits)
        .map_err(|e| format!("invalid base64 ({} digits): {e}", digits.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base64_round_trips_and_matches_rfc_4648() {
        for (raw, b64) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(encode(raw.as_bytes()), b64);
            assert_eq!(decode(b64).unwrap(), raw.as_bytes());
        }
        assert_eq!(decode("Zm9v\nYg").unwrap(), b"foob");
        // Padding is optional.
        assert_eq!(decode("Zg").unwrap(), b"f");
        assert!(decode("Z").is_err());
        assert!(decode("Zm9v!").is_err());
        // Non-zero trailing bits do not encode whole bytes.
        assert!(decode("Zh==").is_err());
    }
}
