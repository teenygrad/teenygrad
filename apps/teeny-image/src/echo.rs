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

//! [`EchoImageModel`], the reference [`ImageModel`].

use futures::future::BoxFuture;

use crate::{
    Background, ImageCapabilities, ImageEvent, ImageInputs, ImageModel, ImageQuality, ImageRequest,
    ImageStream, ImageTask, ImageUsage, ModelInfo, Result, Rgba,
};

/// An image model without weights. It paints the prompt as a colour: a generation is
/// filled with it, an edit paints it into the first image where the mask is transparent
/// (everywhere, without a mask), and a variation inverts the image's colours. A
/// transparent background leaves the left half clear. Image `i` of `n` is the colour
/// darkened by `i`/`n`, so the images differ; partial images fill the top rows first.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoImageModel;

impl EchoImageModel {
    pub const MODEL_ID: &'static str = "teeny-echo-image";
}

/// The colour a prompt names: its FNV-1a hash, opaque.
pub(crate) fn colour(prompt: &str) -> [u8; 4] {
    let h = prompt.bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
    });
    let [r, g, b, _] = h.to_le_bytes();
    [r, g, b, 255]
}

fn shade(c: [u8; 4], i: u32, n: u32) -> [u8; 4] {
    let keep = |v: u8| (u32::from(v) * (n - i) / n) as u8;
    [keep(c[0]), keep(c[1]), keep(c[2]), c[3]]
}

fn paint_one(request: &ImageRequest, inputs: &ImageInputs, i: u32) -> Rgba {
    let (w, h) = request.size.unwrap_or((256, 256));
    let c = shade(
        colour(request.prompt.as_deref().unwrap_or_default()),
        i,
        request.n,
    );
    let mut img = match request.task {
        ImageTask::Generate => Rgba::filled(w, h, c),
        ImageTask::Edit => {
            let mut img = inputs.images[0].clone();
            for (p, px) in img.pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                let open = inputs
                    .mask
                    .as_ref()
                    .is_none_or(|m| m.pixels[p * 4 + 3] == 0);
                if open {
                    px.copy_from_slice(&c);
                }
            }
            img
        }
        ImageTask::Variation => {
            let mut img = inputs.images[0].clone();
            for px in img.pixels.as_chunks_mut::<4>().0 {
                for v in &mut px[..3] {
                    *v = 255 - *v;
                }
            }
            img
        }
    };
    if request.background == Background::Transparent {
        let half = img.width / 2;
        for (p, px) in img.pixels.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            if (p as u32 % img.width) < half {
                px[3] = 0;
            }
        }
    }
    img
}

/// `img` with only its first `k` of `of` row bands painted, the rest blank.
fn partial(img: &Rgba, k: u32, of: u32) -> Rgba {
    let rows = (img.height * k / of) as usize;
    let mut p = Rgba::filled(img.width, img.height, [0; 4]);
    let n = rows * img.width as usize * 4;
    p.pixels[..n].copy_from_slice(&img.pixels[..n]);
    p
}

impl ImageModel for EchoImageModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> ImageCapabilities {
        ImageCapabilities {
            sizes: vec![(256, 256), (512, 512), (1024, 1024)],
            qualities: vec![ImageQuality::Low, ImageQuality::Medium, ImageQuality::High],
            styles: false,
            max_n: 4,
            edits: true,
            max_edit_images: 4,
            masks: true,
            variations: true,
            transparency: true,
            input_fidelity: true,
            max_partial_images: 3,
            streaming: true,
            max_prompt_chars: 32_000,
        }
    }

    fn paint(
        &self,
        request: ImageRequest,
        inputs: ImageInputs,
    ) -> BoxFuture<'_, Result<ImageStream>> {
        Box::pin(async move {
            let mut events = Vec::new();
            for i in 0..request.n {
                let img = paint_one(&request, &inputs, i);
                for k in 0..request.partial_images {
                    events.push(Ok(ImageEvent::Partial {
                        index: i,
                        partial: k,
                        image: partial(&img, k + 1, request.partial_images + 1),
                    }));
                }
                events.push(Ok(ImageEvent::Done {
                    index: i,
                    image: img,
                    revised_prompt: None,
                }));
            }
            let words = request
                .prompt
                .as_deref()
                .unwrap_or_default()
                .split_whitespace()
                .count();
            events.push(Ok(ImageEvent::Usage(ImageUsage {
                input_text_tokens: u32::try_from(words).unwrap_or(u32::MAX),
                input_image_tokens: u32::try_from(inputs.images.len()).unwrap_or(u32::MAX),
                output_tokens: request.n,
            })));
            let stream: ImageStream = Box::pin(futures::stream::iter(events));
            Ok(stream)
        })
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;

    async fn paint(request: ImageRequest, inputs: ImageInputs) -> Vec<ImageEvent<Rgba>> {
        EchoImageModel
            .paint(request, inputs)
            .await
            .unwrap()
            .map(|e| e.unwrap())
            .collect()
            .await
    }

    #[tokio::test]
    async fn test_echo_paints_edits_and_varies() {
        let events = paint(
            ImageRequest {
                prompt: Some("red".into()),
                n: 2,
                size: Some((4, 2)),
                partial_images: 1,
                stream: true,
                ..ImageRequest::default()
            },
            ImageInputs::default(),
        )
        .await;
        assert_eq!(events.len(), 5);
        let ImageEvent::Done { image, .. } = &events[1] else {
            panic!("done second")
        };
        assert_eq!(
            (image.width, image.height, image.get(3, 1)),
            (4, 2, colour("red"))
        );
        let ImageEvent::Partial { image, .. } = &events[0] else {
            panic!("partial first")
        };
        assert_eq!((image.get(0, 0)[3], image.get(0, 1)[3]), (255, 0));

        let base = Rgba::filled(2, 1, [10, 10, 10, 255]);
        let mut mask = Rgba::filled(2, 1, [0, 0, 0, 255]);
        mask.pixels[3] = 0;
        let edit = ImageRequest {
            task: ImageTask::Edit,
            prompt: Some("blue".into()),
            ..ImageRequest::default()
        };
        let events = paint(
            edit,
            ImageInputs {
                images: vec![base.clone()],
                mask: Some(mask),
            },
        )
        .await;
        let ImageEvent::Done { image, .. } = &events[0] else {
            panic!("done")
        };
        assert_eq!(
            (image.get(0, 0), image.get(1, 0)),
            (colour("blue"), [10, 10, 10, 255])
        );

        let vary = ImageRequest {
            task: ImageTask::Variation,
            ..ImageRequest::default()
        };
        let events = paint(
            vary,
            ImageInputs {
                images: vec![base],
                mask: None,
            },
        )
        .await;
        let ImageEvent::Done { image, .. } = &events[0] else {
            panic!("done")
        };
        assert_eq!(image.get(0, 0), [245, 245, 245, 255]);
    }
}
