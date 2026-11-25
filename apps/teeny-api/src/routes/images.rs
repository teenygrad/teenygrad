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

//! The image endpoints: generations, edits and variations.

use axum::{
    Json, Router,
    extract::{
        DefaultBodyLimit, Multipart, State, multipart::MultipartRejection, rejection::JsonRejection,
    },
    response::{IntoResponse, Response, sse::Event},
    routing::post,
};
use futures::{StreamExt, stream};
use teeny_llm::images::{self, ImageEvent, ImageRequest, ImageTask};

use super::{AppState, read_form, sse_error, sse_event, sse_only, unix_now};
use crate::openai::{
    error::ApiError,
    images::{ImageParams, ImageStreamEvent, ImagesResponse, parse_form as parse_image_form},
};

/// The largest image form the edit and variation endpoints take: OpenAI's 50 MiB a file,
/// for a few files.
const MAX_IMAGE_BYTES: usize = 128 * 1024 * 1024;

pub(super) fn routes(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/v1/images/generations", post(image_generations))
        .route(
            "/v1/images/edits",
            post(image_edits).layer(DefaultBodyLimit::max(MAX_IMAGE_BYTES)),
        )
        .route(
            "/v1/images/variations",
            post(image_variations).layer(DefaultBodyLimit::max(MAX_IMAGE_BYTES)),
        )
}

// ---- images ----------------------------------------------------------------------------------

async fn image_generations(
    State(engine): State<AppState>,
    body: Result<Json<ImageParams>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(params) = body?;
    paint(
        engine,
        params.into_core(ImageTask::Generate, Vec::new(), None)?,
    )
    .await
}

async fn image_edits(
    State(engine): State<AppState>,
    form: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let request = parse_image_form(ImageTask::Edit, read_form(form?).await?)?;
    paint(engine, request).await
}

async fn image_variations(
    State(engine): State<AppState>,
    form: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    let request = parse_image_form(ImageTask::Variation, read_form(form?).await?)?;
    paint(engine, request).await
}

async fn paint(engine: AppState, request: ImageRequest) -> Result<Response, ApiError> {
    // What the response describes, without the uploads.
    let described = ImageRequest {
        images: Vec::new(),
        mask: None,
        ..request.clone()
    };
    let created = unix_now();
    let painted = engine.paint(request).await?;
    if !described.stream {
        let (data, usage) = images::collect(painted).await?;
        return Ok(Json(ImagesResponse::new(created, &described, data, usage)).into_response());
    }
    // A completed event carries the usage, which the model sends last: finished images
    // wait for it. Image streams end with their completed events, without `[DONE]`.
    let events = painted
        .scan(Vec::new(), move |pending: &mut Vec<Vec<u8>>, event| {
            let out: Vec<Event> = match event {
                Ok(ImageEvent::Partial { partial, image, .. }) => {
                    vec![sse_event(&ImageStreamEvent::partial(
                        &described, created, partial, &image,
                    ))]
                }
                Ok(ImageEvent::Done { image, .. }) => {
                    pending.push(image);
                    Vec::new()
                }
                Ok(ImageEvent::Usage(usage)) => pending
                    .drain(..)
                    .map(|image| {
                        sse_event(&ImageStreamEvent::completed(
                            &described, created, &image, usage,
                        ))
                    })
                    .collect(),
                Err(e) => vec![sse_error(e)],
            };
            futures::future::ready(Some(stream::iter(out)))
        })
        .flatten();
    Ok(sse_only(events))
}
