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

//! Audio shared by the speech APIs: [`Pcm`] samples, decoding uploaded containers into it
//! (for transcription) and encoding it into response formats (for speech), so no model
//! repeats either.
//!
//! WAV (16-bit integer or 32-bit float) and raw 16-bit PCM are handled in-crate; the
//! compressed containers are recognised and refused by name until a decoder or encoder is
//! added for them.

use crate::{Error, Result};

/// A container an uploaded recording comes in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioContainer {
    Flac,
    Mp3,
    Mp4,
    Mpeg,
    Mpga,
    M4a,
    Ogg,
    Wav,
    Webm,
}

impl AudioContainer {
    /// The container a file name's extension names.
    pub fn from_extension(name: &str) -> Option<Self> {
        let ext = name.rsplit_once('.')?.1.to_ascii_lowercase();
        Some(match ext.as_str() {
            "flac" => Self::Flac,
            "mp3" => Self::Mp3,
            "mp4" => Self::Mp4,
            "mpeg" => Self::Mpeg,
            "mpga" => Self::Mpga,
            "m4a" => Self::M4a,
            "ogg" | "oga" => Self::Ogg,
            "wav" => Self::Wav,
            "webm" => Self::Webm,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::Mp4 => "mp4",
            Self::Mpeg => "mpeg",
            Self::Mpga => "mpga",
            Self::M4a => "m4a",
            Self::Ogg => "ogg",
            Self::Wav => "wav",
            Self::Webm => "webm",
        }
    }
}

/// A format speech can be returned in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioEncoding {
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    /// Raw 16-bit little-endian samples, no header.
    Pcm,
}

impl AudioEncoding {
    pub fn name(self) -> &'static str {
        match self {
            Self::Mp3 => "mp3",
            Self::Opus => "opus",
            Self::Aac => "aac",
            Self::Flac => "flac",
            Self::Wav => "wav",
            Self::Pcm => "pcm",
        }
    }

    /// The media type a response in this format is served as.
    pub fn mime(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/opus",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }

    /// Whether this crate can write the format.
    pub fn encodable(self) -> bool {
        matches!(self, Self::Wav | Self::Pcm)
    }
}

/// Mono or interleaved samples in -1 to 1.
#[derive(Debug, Clone, PartialEq)]
pub struct Pcm {
    pub sample_rate: u32,
    pub channels: u16,
    /// Interleaved by channel.
    pub samples: Vec<f32>,
}

impl Pcm {
    /// The length in seconds.
    pub fn duration(&self) -> f32 {
        if self.sample_rate == 0 || self.channels == 0 {
            return 0.0;
        }
        self.samples.len() as f32 / (f32::from(self.channels) * self.sample_rate as f32)
    }

    /// The channels averaged into one.
    pub fn to_mono(&self) -> Pcm {
        if self.channels <= 1 {
            return self.clone();
        }
        let c = usize::from(self.channels);
        Pcm {
            sample_rate: self.sample_rate,
            channels: 1,
            samples: self
                .samples
                .chunks(c)
                .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
                .collect(),
        }
    }

    /// Mono samples at `rate`, by linear interpolation: what speech models take.
    pub fn resampled_mono(&self, rate: u32) -> Pcm {
        let mono = self.to_mono();
        if mono.sample_rate == rate || mono.samples.is_empty() || mono.sample_rate == 0 {
            return Pcm {
                sample_rate: rate,
                ..mono
            };
        }
        let ratio = f64::from(mono.sample_rate) / f64::from(rate);
        let n = (mono.samples.len() as f64 / ratio).floor() as usize;
        let last = mono.samples.len() - 1;
        let samples = (0..n)
            .map(|i| {
                let at = i as f64 * ratio;
                let lo = (at.floor() as usize).min(last);
                let hi = (lo + 1).min(last);
                let t = (at - lo as f64) as f32;
                mono.samples[lo] * (1.0 - t) + mono.samples[hi] * t
            })
            .collect();
        Pcm {
            sample_rate: rate,
            channels: 1,
            samples,
        }
    }
}

fn unsupported_container(c: AudioContainer) -> Error {
    Error::UnsupportedParameter {
        model: "audio decoder".into(),
        parameter: "file".into(),
        detail: format!("{} audio (only wav is decoded so far)", c.name()),
    }
}

fn bad_file(detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: "file".into(),
        detail: detail.into(),
    }
}

/// Decodes a recording into samples.
pub fn decode(container: AudioContainer, bytes: &[u8]) -> Result<Pcm> {
    match container {
        AudioContainer::Wav => decode_wav(bytes),
        other => Err(unsupported_container(other)),
    }
}

fn le_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Decodes a RIFF WAV file of 16-bit integer or 32-bit float samples.
pub fn decode_wav(bytes: &[u8]) -> Result<Pcm> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(bad_file("not a RIFF WAVE file"));
    }
    let mut at = 12;
    let mut format: Option<(u16, u16, u32, u16)> = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let len = le_u32(bytes, at + 4) as usize;
        let body = at + 8;
        let end = body
            .checked_add(len)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| {
                bad_file(format!(
                    "the `{}` chunk runs past the end of the file",
                    String::from_utf8_lossy(id)
                ))
            })?;
        match id {
            b"fmt " if len >= 16 => {
                format = Some((
                    le_u16(bytes, body),
                    le_u16(bytes, body + 2),
                    le_u32(bytes, body + 4),
                    le_u16(bytes, body + 14),
                ));
            }
            b"data" => {
                let (tag, channels, sample_rate, bits) =
                    format.ok_or_else(|| bad_file("the data chunk comes before the fmt chunk"))?;
                if channels == 0 {
                    return Err(bad_file("the file has no channels"));
                }
                let data = &bytes[body..end];
                let samples = match (tag, bits) {
                    (1, 16) => data
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|s| f32::from(i16::from_le_bytes(*s)) / 32768.0)
                        .collect(),
                    (3, 32) => data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|s| f32::from_le_bytes(*s))
                        .collect(),
                    _ => {
                        return Err(bad_file(format!(
                            "{bits}-bit samples of format {tag}; only 16-bit integer and 32-bit float WAV are read"
                        )));
                    }
                };
                return Ok(Pcm {
                    sample_rate,
                    channels,
                    samples,
                });
            }
            _ => {}
        }
        at = end + (len & 1);
    }
    Err(bad_file("the WAV file has no data chunk"))
}

fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes())
        .collect()
}

/// Encodes samples in `encoding`. Formats this crate cannot write are refused by name.
pub fn encode(pcm: &Pcm, encoding: AudioEncoding) -> Result<Vec<u8>> {
    match encoding {
        AudioEncoding::Pcm => Ok(pcm16(&pcm.samples)),
        AudioEncoding::Wav => Ok(encode_wav(pcm)),
        other => Err(Error::UnsupportedParameter {
            model: "audio encoder".into(),
            parameter: "response_format".into(),
            detail: format!(
                "{} output (only wav and pcm are encoded so far)",
                other.name()
            ),
        }),
    }
}

/// A 16-bit WAV file of `pcm`.
pub fn encode_wav(pcm: &Pcm) -> Vec<u8> {
    let data = pcm16(&pcm.samples);
    let channels = pcm.channels.max(1);
    let block = channels * 2;
    let len = u32::try_from(data.len()).unwrap_or(u32::MAX);
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&len.saturating_add(36).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&pcm.sample_rate.to_le_bytes());
    out.extend_from_slice(&(pcm.sample_rate * u32::from(block)).to_le_bytes());
    out.extend_from_slice(&block.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, seconds: f32) -> Pcm {
        let n = (rate as f32 * seconds) as usize;
        Pcm {
            sample_rate: rate,
            channels: 1,
            samples: (0..n).map(|i| (i as f32 * 0.05).sin() * 0.5).collect(),
        }
    }

    #[test]
    fn test_wav_round_trips_through_encode_and_decode() {
        let pcm = tone(8000, 0.25);
        let back = decode(AudioContainer::Wav, &encode_wav(&pcm)).unwrap();
        assert_eq!(
            (back.sample_rate, back.channels, back.samples.len()),
            (8000, 1, 2000)
        );
        assert!(
            back.samples
                .iter()
                .zip(&pcm.samples)
                .all(|(a, b)| (a - b).abs() < 1e-4)
        );
        assert!((back.duration() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_decode_refuses_what_it_cannot_read_by_name() {
        assert!(matches!(
            decode(AudioContainer::Mp3, b"ID3"),
            Err(Error::UnsupportedParameter { .. })
        ));
        assert!(matches!(
            decode(AudioContainer::Wav, b"not audio"),
            Err(Error::InvalidParameter { .. })
        ));
        assert!(matches!(
            encode(&tone(8000, 0.1), AudioEncoding::Mp3),
            Err(Error::UnsupportedParameter { .. })
        ));
        assert_eq!(
            AudioContainer::from_extension("Talk.WAV"),
            Some(AudioContainer::Wav)
        );
        assert_eq!(AudioContainer::from_extension("notes.txt"), None);
    }

    #[test]
    fn test_resampled_mono_mixes_channels_and_changes_rate() {
        let stereo = Pcm {
            sample_rate: 32000,
            channels: 2,
            samples: [0.2, 0.4].repeat(32000),
        };
        let m = stereo.resampled_mono(16000);
        assert_eq!(
            (m.sample_rate, m.channels, m.samples.len()),
            (16000, 1, 16000)
        );
        assert!(m.samples.iter().all(|s| (s - 0.3).abs() < 1e-6));
        assert_eq!(encode(&m, AudioEncoding::Pcm).unwrap().len(), 32000);
    }
}
