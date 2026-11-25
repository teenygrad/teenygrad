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

//! `teeny-image`: teenygrad's image generation API (`/v1/images/generations`, `/edits`,
//! `/variations`), for diffusion-class models.
//!
//! A model implements [`ImageModel`]: it declares its [`ImageCapabilities`] and paints
//! [`Rgba`] pixels ([`ImageStream`]), from the decoded [`ImageInputs`] of an edit or a
//! variation. Decoding uploads and encoding results ([`decode`], [`encode`],
//! [`encode_stream`]) are shared, so no model repeats them; [`validate`] refuses what the
//! model cannot do first. [`EchoImageModel`] is the reference implementation.
//!
//! PNG is decoded and encoded so far; JPEG and WebP are refused by name. Nothing here knows
//! about HTTP: [`teeny-api`] serves the API with its `image` feature.
//!
//! [`teeny-api`]: https://docs.teenygrad.org/api/teenygrad/teeny_api/

mod echo;
mod error;

pub use echo::EchoImageModel;
pub use error::{Error, Result};

use std::pin::Pin;

use futures::future::BoxFuture;
use futures::{Stream, StreamExt};

/// What a model is, as a frontend lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    /// Unix timestamp (seconds) the model was loaded/created.
    pub created: u64,
    pub owned_by: String,
}

/// An image as 8-bit RGBA, row-major.
#[derive(Clone, PartialEq, Eq)]
pub struct Rgba {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl std::fmt::Debug for Rgba {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rgba({}x{})", self.width, self.height)
    }
}

impl Rgba {
    /// A `width` x `height` image of one colour.
    pub fn filled(width: u32, height: u32, rgba: [u8; 4]) -> Self {
        Self {
            width,
            height,
            pixels: rgba.repeat(width as usize * height as usize),
        }
    }

    /// The pixel at `x`, `y`.
    pub fn get(&self, x: u32, y: u32) -> [u8; 4] {
        let at = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.pixels[at],
            self.pixels[at + 1],
            self.pixels[at + 2],
            self.pixels[at + 3],
        ]
    }
}

/// Image file formats.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageFormat {
    #[default]
    Png,
    Jpeg,
    Webp,
}

impl ImageFormat {
    pub fn name(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Jpeg => "jpeg",
            Self::Webp => "webp",
        }
    }

    /// Whether this crate can read and write the format.
    pub fn supported(self) -> bool {
        self == Self::Png
    }
}

/// Encodes `image` as `format`.
pub fn encode(image: &Rgba, format: ImageFormat) -> Result<Vec<u8>> {
    if format != ImageFormat::Png {
        return Err(Error::UnsupportedParameter {
            model: "image encoder".into(),
            parameter: "output_format".into(),
            detail: format!("{} output (only png is encoded so far)", format.name()),
        });
    }
    let buffer = image::RgbaImage::from_raw(image.width, image.height, image.pixels.clone())
        .ok_or_else(|| {
            Error::Inference(format!(
                "a {}x{} image has {} bytes of pixels, not {}",
                image.width,
                image.height,
                image.pixels.len(),
                image.width as usize * image.height as usize * 4
            ))
        })?;
    let mut out = std::io::Cursor::new(Vec::new());
    buffer
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| Error::Inference(format!("encoding a png: {e}")))?;
    Ok(out.into_inner())
}

/// Decodes an uploaded image (`parameter` names the field, for errors).
pub fn decode(bytes: &[u8], parameter: &str) -> Result<Rgba> {
    let invalid = |detail: String| Error::InvalidParameter {
        parameter: parameter.to_owned(),
        detail,
    };
    match image::guess_format(bytes) {
        Ok(image::ImageFormat::Png) => {}
        Ok(other) => {
            return Err(Error::UnsupportedParameter {
                model: "image decoder".into(),
                parameter: parameter.to_owned(),
                detail: format!("{other:?} images (only png is decoded so far)"),
            });
        }
        Err(_) => return Err(invalid("not an image file".into())),
    }
    let decoded = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .map_err(|e| invalid(format!("an unreadable png: {e}")))?
        .into_rgba8();
    Ok(Rgba {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    })
}

/// How carefully to paint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageQuality {
    #[default]
    Auto,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Standard,
    Hd,
}

impl ImageQuality {
    pub fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::XHigh => "xhigh",
            Self::Max => "max",
            Self::Standard => "standard",
            Self::Hd => "hd",
        }
    }
}

/// A painting style, for models that take one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageStyle {
    Vivid,
    Natural,
}

/// Whether the background may be transparent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Background {
    #[default]
    Auto,
    Transparent,
    Opaque,
}

/// How closely an edit keeps to its input images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFidelity {
    Low,
    High,
}

/// What to paint from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageTask {
    /// From the prompt alone.
    #[default]
    Generate,
    /// The prompt applied to the input images, within the mask if given.
    Edit,
    /// Variations on one input image.
    Variation,
}

/// A request for images.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageRequest {
    pub model: String,
    pub task: ImageTask,
    pub prompt: Option<String>,
    /// The uploads to edit or vary, as sent.
    pub images: Vec<Vec<u8>>,
    pub mask: Option<Vec<u8>>,
    /// How many images, 1 to 10.
    pub n: u32,
    /// Width and height; `None` lets the model choose.
    pub size: Option<(u32, u32)>,
    pub quality: ImageQuality,
    pub style: Option<ImageStyle>,
    pub background: Background,
    pub format: ImageFormat,
    /// 0 to 100, for JPEG and WebP.
    pub compression: Option<u8>,
    pub input_fidelity: Option<InputFidelity>,
    /// Partial images to stream before each finished one, 0 to 3.
    pub partial_images: u32,
    pub stream: bool,
}

impl Default for ImageRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            task: ImageTask::Generate,
            prompt: None,
            images: Vec::new(),
            mask: None,
            n: 1,
            size: None,
            quality: ImageQuality::Auto,
            style: None,
            background: Background::Auto,
            format: ImageFormat::Png,
            compression: None,
            input_fidelity: None,
            partial_images: 0,
            stream: false,
        }
    }
}

/// The decoded uploads of an edit or variation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageInputs {
    pub images: Vec<Rgba>,
    pub mask: Option<Rgba>,
}

/// What painting used.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImageUsage {
    pub input_text_tokens: u32,
    pub input_image_tokens: u32,
    pub output_tokens: u32,
}

/// A model's output, or the same with the images encoded (`T` is [`Rgba`], then bytes).
#[derive(Debug, Clone, PartialEq)]
pub enum ImageEvent<T> {
    /// A partial rendering of image `index`, the `partial`-th.
    Partial {
        index: u32,
        partial: u32,
        image: T,
    },
    /// Image `index`, finished, with the prompt as the model rewrote it if it did.
    Done {
        index: u32,
        image: T,
        revised_prompt: Option<String>,
    },
    Usage(ImageUsage),
}

/// An image model's output.
pub type ImageStream = Pin<Box<dyn Stream<Item = Result<ImageEvent<Rgba>>> + Send>>;

/// Images in the requested format.
pub type EncodedImageStream = Pin<Box<dyn Stream<Item = Result<ImageEvent<Vec<u8>>>> + Send>>;

/// What an image model can do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageCapabilities {
    /// The sizes it paints; the first is its default.
    pub sizes: Vec<(u32, u32)>,
    pub qualities: Vec<ImageQuality>,
    pub styles: bool,
    pub max_n: u32,
    pub edits: bool,
    /// The most images one edit takes.
    pub max_edit_images: usize,
    pub masks: bool,
    pub variations: bool,
    pub transparency: bool,
    pub input_fidelity: bool,
    /// The most partial images it streams per image; 0 if it streams none.
    pub max_partial_images: u32,
    pub streaming: bool,
    /// The longest prompt, in characters.
    pub max_prompt_chars: usize,
}

impl Default for ImageCapabilities {
    fn default() -> Self {
        Self {
            sizes: vec![(1024, 1024)],
            qualities: vec![ImageQuality::Auto],
            styles: false,
            max_n: 1,
            edits: false,
            max_edit_images: 1,
            masks: false,
            variations: false,
            transparency: false,
            input_fidelity: false,
            max_partial_images: 0,
            streaming: false,
            max_prompt_chars: 32_000,
        }
    }
}

fn invalid(parameter: &str, detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: parameter.to_owned(),
        detail: detail.into(),
    }
}

fn unsupported(model: &str, parameter: &str, detail: impl Into<String>) -> Error {
    Error::UnsupportedParameter {
        model: model.to_owned(),
        parameter: parameter.to_owned(),
        detail: detail.into(),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`, before its uploads are
/// decoded; [`check_inputs`] checks them once they are.
pub fn validate(model: &str, caps: &ImageCapabilities, r: &ImageRequest) -> Result<()> {
    match r.task {
        ImageTask::Generate => {}
        ImageTask::Edit if !caps.edits => return Err(unsupported(model, "model", "image edits")),
        ImageTask::Variation if !caps.variations => {
            return Err(unsupported(model, "model", "image variations"));
        }
        ImageTask::Edit | ImageTask::Variation => {
            if r.images.is_empty() {
                return Err(invalid("image", "there is no image to work from"));
            }
        }
    }
    if r.task == ImageTask::Edit && r.images.len() > caps.max_edit_images {
        return Err(unsupported(
            model,
            "image",
            format!(
                "{} input images; it edits at most {}",
                r.images.len(),
                caps.max_edit_images
            ),
        ));
    }
    if r.task == ImageTask::Variation && r.images.len() > 1 {
        return Err(invalid("image", "a variation starts from one image"));
    }
    if r.mask.is_some() && (r.task != ImageTask::Edit || !caps.masks) {
        return Err(unsupported(model, "mask", "masks"));
    }
    if r.task != ImageTask::Variation {
        match r.prompt.as_deref().map(str::trim) {
            None | Some("") => return Err(invalid("prompt", "there is no prompt")),
            Some(p) if p.chars().count() > caps.max_prompt_chars => {
                return Err(invalid(
                    "prompt",
                    format!(
                        "{} characters; the model reads at most {}",
                        p.chars().count(),
                        caps.max_prompt_chars
                    ),
                ));
            }
            Some(_) => {}
        }
    }
    if !(1..=10).contains(&r.n) {
        return Err(invalid("n", format!("{} is outside 1 to 10", r.n)));
    }
    if r.n > caps.max_n {
        return Err(unsupported(
            model,
            "n",
            format!("{} images; it paints at most {}", r.n, caps.max_n),
        ));
    }
    if let Some((w, h)) = r.size
        && !caps.sizes.contains(&(w, h))
    {
        let sizes: Vec<String> = caps.sizes.iter().map(|(w, h)| format!("{w}x{h}")).collect();
        return Err(unsupported(
            model,
            "size",
            format!("{w}x{h}; it paints {}", sizes.join(", ")),
        ));
    }
    if r.quality != ImageQuality::Auto && !caps.qualities.contains(&r.quality) {
        return Err(unsupported(
            model,
            "quality",
            format!("`{}` quality", r.quality.name()),
        ));
    }
    if r.style.is_some() && !caps.styles {
        return Err(unsupported(model, "style", "styles"));
    }
    if r.background == Background::Transparent {
        if !caps.transparency {
            return Err(unsupported(model, "background", "transparent backgrounds"));
        }
        if r.format == ImageFormat::Jpeg {
            return Err(invalid(
                "background",
                "jpeg has no transparency; use png or webp",
            ));
        }
    }
    if !r.format.supported() {
        return Err(unsupported(
            model,
            "output_format",
            format!("{} output (only png is encoded so far)", r.format.name()),
        ));
    }
    if let Some(c) = r.compression {
        if c > 100 {
            return Err(invalid(
                "output_compression",
                format!("{c} is outside 0 to 100"),
            ));
        }
        if r.format == ImageFormat::Png {
            return Err(invalid(
                "output_compression",
                "png is lossless; compression applies to jpeg and webp",
            ));
        }
    }
    if r.input_fidelity.is_some() && (r.task != ImageTask::Edit || !caps.input_fidelity) {
        return Err(unsupported(model, "input_fidelity", "input fidelity"));
    }
    if r.partial_images > 3 {
        return Err(invalid(
            "partial_images",
            format!("{} is outside 0 to 3", r.partial_images),
        ));
    }
    if r.partial_images > 0 && !r.stream {
        return Err(invalid(
            "partial_images",
            "partial images are only sent when streaming",
        ));
    }
    if r.partial_images > caps.max_partial_images {
        return Err(unsupported(
            model,
            "partial_images",
            format!(
                "{} partial images; it streams at most {}",
                r.partial_images, caps.max_partial_images
            ),
        ));
    }
    if r.stream && !caps.streaming {
        return Err(unsupported(model, "stream", "streamed images"));
    }
    Ok(())
}

/// Checks decoded uploads: a mask must match the first image's size.
pub fn check_inputs(inputs: &ImageInputs) -> Result<()> {
    if let (Some(mask), Some(first)) = (&inputs.mask, inputs.images.first())
        && (mask.width, mask.height) != (first.width, first.height)
    {
        return Err(invalid(
            "mask",
            format!(
                "the mask is {}x{} but the image is {}x{}",
                mask.width, mask.height, first.width, first.height
            ),
        ));
    }
    Ok(())
}

/// Decodes a request's uploads.
pub fn decode_inputs(r: &ImageRequest) -> Result<ImageInputs> {
    let inputs = ImageInputs {
        images: r
            .images
            .iter()
            .map(|b| decode(b, "image"))
            .collect::<Result<_>>()?,
        mask: r.mask.as_deref().map(|b| decode(b, "mask")).transpose()?,
    };
    check_inputs(&inputs)?;
    Ok(inputs)
}

/// A model that paints images.
pub trait ImageModel: Send + Sync + 'static {
    fn info(&self) -> ModelInfo;

    fn capabilities(&self) -> ImageCapabilities;

    /// Paints `request`, which has passed [`validate`], from `inputs` (empty for a
    /// generation). The stream carries each of the `n` images, finished, and ends with
    /// [`ImageEvent::Usage`].
    fn paint(
        &self,
        request: ImageRequest,
        inputs: ImageInputs,
    ) -> BoxFuture<'_, Result<ImageStream>>;
}

/// Encodes a model's images as `format`, as they come.
pub fn encode_stream(stream: ImageStream, format: ImageFormat) -> EncodedImageStream {
    Box::pin(stream.map(move |event| {
        Ok(match event? {
            ImageEvent::Partial {
                index,
                partial,
                image,
            } => ImageEvent::Partial {
                index,
                partial,
                image: encode(&image, format)?,
            },
            ImageEvent::Done {
                index,
                image,
                revised_prompt,
            } => ImageEvent::Done {
                index,
                image: encode(&image, format)?,
                revised_prompt,
            },
            ImageEvent::Usage(u) => ImageEvent::Usage(u),
        })
    }))
}

/// The finished images of a stream, by index, with their revised prompts, and its usage.
pub async fn collect(
    mut stream: EncodedImageStream,
) -> Result<(Vec<(Vec<u8>, Option<String>)>, ImageUsage)> {
    let mut images: Vec<(u32, Vec<u8>, Option<String>)> = Vec::new();
    let mut usage = ImageUsage::default();
    while let Some(event) = stream.next().await {
        match event? {
            ImageEvent::Partial { .. } => {}
            ImageEvent::Done {
                index,
                image,
                revised_prompt,
            } => images.push((index, image, revised_prompt)),
            ImageEvent::Usage(u) => usage = u,
        }
    }
    images.sort_by_key(|(i, ..)| *i);
    Ok((images.into_iter().map(|(_, b, p)| (b, p)).collect(), usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> ImageCapabilities {
        ImageCapabilities {
            sizes: vec![(256, 256)],
            max_n: 2,
            ..ImageCapabilities::default()
        }
    }

    fn request() -> ImageRequest {
        ImageRequest {
            model: "m".into(),
            prompt: Some("a cat".into()),
            ..ImageRequest::default()
        }
    }

    fn param(r: Result<()>) -> String {
        match r {
            Err(Error::InvalidParameter { parameter, .. })
            | Err(Error::UnsupportedParameter { parameter, .. }) => parameter,
            other => panic!("not refused by parameter: {other:?}"),
        }
    }

    #[test]
    fn test_validate_refuses_what_the_model_cannot_do() {
        assert!(validate("m", &caps(), &request()).is_ok());
        let cases = [
            (
                ImageRequest {
                    prompt: None,
                    ..request()
                },
                "prompt",
            ),
            (ImageRequest { n: 11, ..request() }, "n"),
            (ImageRequest { n: 3, ..request() }, "n"),
            (
                ImageRequest {
                    size: Some((512, 512)),
                    ..request()
                },
                "size",
            ),
            (
                ImageRequest {
                    quality: ImageQuality::Hd,
                    ..request()
                },
                "quality",
            ),
            (
                ImageRequest {
                    style: Some(ImageStyle::Vivid),
                    ..request()
                },
                "style",
            ),
            (
                ImageRequest {
                    background: Background::Transparent,
                    ..request()
                },
                "background",
            ),
            (
                ImageRequest {
                    format: ImageFormat::Webp,
                    ..request()
                },
                "output_format",
            ),
            (
                ImageRequest {
                    compression: Some(50),
                    ..request()
                },
                "output_compression",
            ),
            (
                ImageRequest {
                    partial_images: 2,
                    ..request()
                },
                "partial_images",
            ),
            (
                ImageRequest {
                    stream: true,
                    ..request()
                },
                "stream",
            ),
            (
                ImageRequest {
                    task: ImageTask::Edit,
                    ..request()
                },
                "model",
            ),
            (
                ImageRequest {
                    task: ImageTask::Variation,
                    ..request()
                },
                "model",
            ),
            (
                ImageRequest {
                    mask: Some(vec![1]),
                    ..request()
                },
                "mask",
            ),
        ];
        for (r, want) in cases {
            assert_eq!(param(validate("m", &caps(), &r)), want);
        }
    }

    #[test]
    fn test_png_round_trips_and_other_formats_are_refused() {
        let img = Rgba::filled(3, 2, [10, 20, 30, 255]);
        let png = encode(&img, ImageFormat::Png).unwrap();
        assert_eq!(decode(&png, "image").unwrap(), img);
        assert!(matches!(
            encode(&img, ImageFormat::Jpeg),
            Err(Error::UnsupportedParameter { .. })
        ));
        assert!(matches!(
            decode(b"\xff\xd8\xff\xe0 jpeg", "image"),
            Err(Error::UnsupportedParameter { .. })
        ));
        assert!(
            matches!(decode(b"text", "mask"), Err(Error::InvalidParameter { ref parameter, .. }) if parameter == "mask")
        );
        let mismatched = ImageInputs {
            images: vec![img],
            mask: Some(Rgba::filled(1, 1, [0; 4])),
        };
        assert_eq!(param(check_inputs(&mismatched)), "mask");
    }
}
