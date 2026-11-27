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

//! Decoding the base64 clients send inline (chat's audio and file data), with the
//! `base64` crate: the standard alphabet, whitespace ignored (line-wrapped data), padding
//! optional. Encoding needs no such leniency, so callers use `BASE64_STANDARD` directly.

use base64::Engine as _;
use base64::alphabet::STANDARD;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};

/// The standard alphabet, taking input with or without padding.
const LENIENT: GeneralPurpose = GeneralPurpose::new(
    &STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// Decodes `text`, ignoring whitespace; padding is optional.
pub(crate) fn decode(text: &str) -> Result<Vec<u8>, String> {
    let digits: Vec<u8> = text.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    LENIENT
        .decode(&digits)
        .map_err(|e| format!("invalid base64 ({} digits): {e}", digits.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::prelude::BASE64_STANDARD;

    #[test]
    fn test_decode_is_lenient_about_padding_and_whitespace() {
        for (raw, b64) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(BASE64_STANDARD.encode(raw), b64);
            assert_eq!(decode(b64).unwrap(), raw.as_bytes());
        }
        assert_eq!(decode("Zg").unwrap(), b"f");
        assert_eq!(decode("Zm9v\nYg").unwrap(), b"foob");
        assert!(decode("Z").is_err());
        assert!(decode("Zm9v!").is_err());
        // Non-zero trailing bits do not encode whole bytes.
        assert!(decode("Zh==").is_err());
    }
}
