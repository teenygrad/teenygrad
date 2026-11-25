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

//! `/v1/images/generations` (JSON), `/edits` and `/variations` (multipart forms): the
//! requests, `ImagesResponse`, and the streamed image events.

use serde::{Deserialize, Serialize};
use teeny_llm_core::images::{
    Background, ImageFormat, ImageQuality, ImageRequest, ImageStyle, ImageTask, ImageUsage,
    InputFidelity,
};

use super::audio::FormPart;
use super::base64;
use super::error::ApiError;

/// The parameters the three endpoints share, as sent; `CreateImageRequest` for
/// generations, and what the edit and variation forms are read into.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ImageParams {
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub size: Option<String>,
    #[serde(default)]
    pub quality: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub output_format: Option<String>,
    #[serde(default)]
    pub output_compression: Option<u8>,
    #[serde(default)]
    pub moderation: Option<String>,
    #[serde(default)]
    pub input_fidelity: Option<String>,
    #[serde(default)]
    pub partial_images: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub user: Option<String>,
}

fn bad(param: &str, detail: String) -> ApiError {
    ApiError::invalid_request(detail, Some(param))
}

fn choose<T: Copy>(
    param: &str,
    value: Option<&str>,
    default: T,
    options: &[(&str, T)],
) -> Result<T, ApiError> {
    let Some(v) = value else { return Ok(default) };
    options
        .iter()
        .find(|(name, _)| *name == v)
        .map(|(_, t)| *t)
        .ok_or_else(|| {
            let names: Vec<&str> = options.iter().map(|(n, _)| *n).collect();
            bad(param, format!("`{v}` is not one of {}", names.join(", ")))
        })
}

fn size(value: Option<&str>) -> Result<Option<(u32, u32)>, ApiError> {
    match value {
        None | Some("auto") => Ok(None),
        Some(s) => s
            .split_once('x')
            .and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))
            .filter(|&(w, h)| w > 0 && h > 0)
            .map(Some)
            .ok_or_else(|| {
                bad(
                    "size",
                    format!("`{s}` is not a size; use auto or WIDTHxHEIGHT"),
                )
            }),
    }
}

const QUALITIES: [(&str, ImageQuality); 8] = [
    ("auto", ImageQuality::Auto),
    ("low", ImageQuality::Low),
    ("medium", ImageQuality::Medium),
    ("high", ImageQuality::High),
    ("xhigh", ImageQuality::XHigh),
    ("max", ImageQuality::Max),
    ("standard", ImageQuality::Standard),
    ("hd", ImageQuality::Hd),
];

const BACKGROUNDS: [(&str, Background); 3] = [
    ("auto", Background::Auto),
    ("transparent", Background::Transparent),
    ("opaque", Background::Opaque),
];

const FORMATS: [(&str, ImageFormat); 3] = [
    ("png", ImageFormat::Png),
    ("jpeg", ImageFormat::Jpeg),
    ("webp", ImageFormat::Webp),
];

impl ImageParams {
    /// The core request for `task`, with the uploads of an edit or variation.
    pub fn into_core(
        self,
        task: ImageTask,
        images: Vec<Vec<u8>>,
        mask: Option<Vec<u8>>,
    ) -> Result<ImageRequest, ApiError> {
        if let Some(f) = self.response_format.as_deref()
            && f != "b64_json"
        {
            return Err(bad(
                "response_format",
                format!(
                    "`{f}`: this server returns images as b64_json; it does not keep them to serve URLs"
                ),
            ));
        }
        choose(
            "moderation",
            self.moderation.as_deref(),
            (),
            &[("auto", ()), ("low", ())],
        )?;
        let style = choose(
            "style",
            self.style.as_deref(),
            None,
            &[
                ("vivid", Some(ImageStyle::Vivid)),
                ("natural", Some(ImageStyle::Natural)),
            ],
        )?;
        let input_fidelity = choose(
            "input_fidelity",
            self.input_fidelity.as_deref(),
            None,
            &[
                ("low", Some(InputFidelity::Low)),
                ("high", Some(InputFidelity::High)),
            ],
        )?;
        Ok(ImageRequest {
            model: self.model,
            task,
            prompt: self.prompt,
            images,
            mask,
            n: self.n.unwrap_or(1),
            size: size(self.size.as_deref())?,
            quality: choose(
                "quality",
                self.quality.as_deref(),
                ImageQuality::Auto,
                &QUALITIES,
            )?,
            style,
            background: choose(
                "background",
                self.background.as_deref(),
                Background::Auto,
                &BACKGROUNDS,
            )?,
            format: choose(
                "output_format",
                self.output_format.as_deref(),
                ImageFormat::Png,
                &FORMATS,
            )?,
            compression: self.output_compression,
            input_fidelity,
            partial_images: self.partial_images.unwrap_or(0),
            stream: self.stream.unwrap_or(false),
        })
    }
}

fn text(part: &FormPart) -> Result<String, ApiError> {
    String::from_utf8(part.data.clone()).map_err(|_| {
        bad(
            &part.name,
            format!("the `{}` field is not UTF-8 text", part.name),
        )
    })
}

fn number<T: std::str::FromStr>(part: &FormPart) -> Result<T, ApiError> {
    let t = text(part)?;
    t.trim()
        .parse()
        .map_err(|_| bad(&part.name, format!("`{t}` is not a whole number")))
}

/// The core request an edit or variation form describes.
pub fn parse_form(task: ImageTask, parts: Vec<FormPart>) -> Result<ImageRequest, ApiError> {
    let mut p = ImageParams::default();
    let mut images = Vec::new();
    let mut mask = None;
    let mut have_model = false;
    for part in parts {
        let name = part.name.trim_end_matches("[]").to_owned();
        let edit = task == ImageTask::Edit;
        match name.as_str() {
            "image" => images.push(part.data),
            "mask" if edit => mask = Some(part.data),
            "model" => {
                p.model = text(&part)?;
                have_model = true;
            }
            "n" => p.n = Some(number(&part)?),
            "size" => p.size = Some(text(&part)?),
            "response_format" => p.response_format = Some(text(&part)?),
            "user" => p.user = Some(text(&part)?),
            "prompt" if edit => p.prompt = Some(text(&part)?),
            "quality" if edit => p.quality = Some(text(&part)?),
            "background" if edit => p.background = Some(text(&part)?),
            "output_format" if edit => p.output_format = Some(text(&part)?),
            "output_compression" if edit => p.output_compression = Some(number(&part)?),
            "input_fidelity" if edit => p.input_fidelity = Some(text(&part)?),
            "partial_images" if edit => p.partial_images = Some(number(&part)?),
            "stream" if edit => {
                p.stream = Some(match text(&part)?.trim() {
                    "true" | "1" => true,
                    "false" | "0" => false,
                    other => return Err(bad("stream", format!("`{other}` is not a boolean"))),
                });
            }
            // Fields this server does not act on are ignored, as for JSON bodies.
            _ => {}
        }
    }
    if !have_model {
        return Err(bad("model", "the form has no `model` field".into()));
    }
    if images.is_empty() {
        return Err(bad("image", "the form has no `image` file".into()));
    }
    p.into_core(task, images, mask)
}

/// `ImagesUsage`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ImagesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub input_tokens_details: InputTokensDetails,
}

/// The split of `ImagesUsage`'s input tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct InputTokensDetails {
    pub text_tokens: u32,
    pub image_tokens: u32,
}

impl From<ImageUsage> for ImagesUsage {
    fn from(u: ImageUsage) -> Self {
        let input = u.input_text_tokens.saturating_add(u.input_image_tokens);
        Self {
            input_tokens: input,
            output_tokens: u.output_tokens,
            total_tokens: input.saturating_add(u.output_tokens),
            input_tokens_details: InputTokensDetails {
                text_tokens: u.input_text_tokens,
                image_tokens: u.input_image_tokens,
            },
        }
    }
}

/// One image of `ImagesResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageData {
    pub b64_json: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

/// `ImagesResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImagesResponse {
    pub created: u64,
    pub data: Vec<ImageData>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<&'static str>,
    pub output_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    pub usage: ImagesUsage,
}

fn background_name(b: Background) -> &'static str {
    match b {
        Background::Auto => "auto",
        Background::Transparent => "transparent",
        Background::Opaque => "opaque",
    }
}

fn size_name(r: &ImageRequest) -> String {
    r.size
        .map_or_else(|| "auto".to_owned(), |(w, h)| format!("{w}x{h}"))
}

impl ImagesResponse {
    /// The response for `request`, painted as `images` with `usage`.
    pub fn new(
        created: u64,
        request: &ImageRequest,
        images: Vec<(Vec<u8>, Option<String>)>,
        usage: ImageUsage,
    ) -> Self {
        Self {
            created,
            data: images
                .into_iter()
                .map(|(bytes, revised_prompt)| ImageData {
                    b64_json: base64::encode(&bytes),
                    revised_prompt,
                })
                .collect(),
            background: (request.background != Background::Auto)
                .then(|| background_name(request.background)),
            output_format: request.format.name(),
            quality: (request.quality != ImageQuality::Auto).then(|| request.quality.name()),
            size: request.size.map(|_| size_name(request)),
            usage: usage.into(),
        }
    }
}

/// `ImageGenPartialImageEvent`, `ImageGenCompletedEvent` and their edit twins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ImageStreamEvent {
    #[serde(rename = "type")]
    pub kind: String,
    pub b64_json: String,
    pub created_at: u64,
    pub size: String,
    pub quality: &'static str,
    pub background: &'static str,
    pub output_format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partial_image_index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImagesUsage>,
}

impl ImageStreamEvent {
    fn new(request: &ImageRequest, created_at: u64, suffix: &str, bytes: &[u8]) -> Self {
        let prefix = if request.task == ImageTask::Edit {
            "image_edit"
        } else {
            "image_generation"
        };
        Self {
            kind: format!("{prefix}.{suffix}"),
            b64_json: base64::encode(bytes),
            created_at,
            size: size_name(request),
            quality: request.quality.name(),
            background: background_name(request.background),
            output_format: request.format.name(),
            partial_image_index: None,
            usage: None,
        }
    }

    /// A partial image, the `index`-th.
    pub fn partial(request: &ImageRequest, created_at: u64, index: u32, bytes: &[u8]) -> Self {
        Self {
            partial_image_index: Some(index),
            ..Self::new(request, created_at, "partial_image", bytes)
        }
    }

    /// A finished image, with the usage so far.
    pub fn completed(
        request: &ImageRequest,
        created_at: u64,
        bytes: &[u8],
        usage: ImageUsage,
    ) -> Self {
        Self {
            usage: Some(usage.into()),
            ..Self::new(request, created_at, "completed", bytes)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(name: &str, value: &str) -> FormPart {
        FormPart {
            name: name.into(),
            file_name: None,
            data: value.as_bytes().to_vec(),
        }
    }

    #[test]
    fn test_params_map_to_the_core_request() {
        let p: ImageParams = serde_json::from_value(serde_json::json!({
            "model": "m", "prompt": "a cat", "n": 2, "size": "512x512", "quality": "high",
            "background": "transparent", "output_format": "png", "moderation": "low"
        }))
        .unwrap();
        let r = p.into_core(ImageTask::Generate, Vec::new(), None).unwrap();
        assert_eq!(
            (r.n, r.size, r.quality),
            (2, Some((512, 512)), ImageQuality::High)
        );
        assert_eq!(r.background, Background::Transparent);
        let refuse = |v: serde_json::Value| {
            let p: ImageParams = serde_json::from_value(v).unwrap();
            p.into_core(ImageTask::Generate, Vec::new(), None)
                .unwrap_err()
                .body
                .param
        };
        assert_eq!(
            refuse(serde_json::json!({"model": "m", "size": "big"})).as_deref(),
            Some("size")
        );
        assert_eq!(
            refuse(serde_json::json!({"model": "m", "response_format": "url"})).as_deref(),
            Some("response_format")
        );
        assert_eq!(
            refuse(serde_json::json!({"model": "m", "quality": "ultra"})).as_deref(),
            Some("quality")
        );
    }

    #[test]
    fn test_parse_form_reads_edits_and_variations() {
        let file = |name: &str| FormPart {
            name: name.into(),
            file_name: Some("x.png".into()),
            data: vec![1],
        };
        let r = parse_form(
            ImageTask::Edit,
            vec![
                file("image[]"),
                file("image[]"),
                file("mask"),
                part("model", "m"),
                part("prompt", "p"),
                part("n", "2"),
            ],
        )
        .unwrap();
        assert_eq!(
            (r.images.len(), r.mask.is_some(), r.n, r.prompt.as_deref()),
            (2, true, 2, Some("p"))
        );
        // A variation takes no prompt or mask; they are ignored.
        let r = parse_form(
            ImageTask::Variation,
            vec![
                file("image"),
                file("mask"),
                part("model", "m"),
                part("prompt", "p"),
            ],
        )
        .unwrap();
        assert_eq!((r.mask, r.prompt), (None, None));
        let missing = parse_form(ImageTask::Variation, vec![part("model", "m")]);
        assert_eq!(missing.unwrap_err().body.param.as_deref(), Some("image"));
        let bad_n = parse_form(
            ImageTask::Variation,
            vec![file("image"), part("model", "m"), part("n", "two")],
        );
        assert_eq!(bad_n.unwrap_err().body.param.as_deref(), Some("n"));
    }
}
