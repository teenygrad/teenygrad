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

//! The audio endpoints: transcriptions, translations and speech.

use axum::{
    Json, Router,
    extract::{
        DefaultBodyLimit, Multipart, State, multipart::MultipartRejection, rejection::JsonRejection,
    },
    http::header,
    response::{IntoResponse, Response},
    routing::post,
};
use futures::StreamExt;
use teeny_audio::speech::{self, SpeechChunk};
use teeny_audio::transcription::{self, TranscriptionEvent, TranscriptionTask};

use super::{AppState, read_form, sse_error, sse_event, sse_only};
use crate::openai::{
    audio::{Rendered, TextDeltaEvent, TextDoneEvent, parse_form, render},
    error::ApiError,
    speech::{AudioDeltaEvent, AudioDoneEvent, CreateSpeechRequest, StreamFormat},
};

/// The largest upload the audio endpoints take, as OpenAI's do (25 MiB), plus room for the
/// form's other fields.
const MAX_AUDIO_BYTES: usize = 26 * 1024 * 1024;

pub(super) fn routes(router: Router<AppState>) -> Router<AppState> {
    router
        .route("/v1/audio/speech", post(speech_audio))
        .route(
            "/v1/audio/transcriptions",
            post(transcriptions).layer(DefaultBodyLimit::max(MAX_AUDIO_BYTES)),
        )
        .route(
            "/v1/audio/translations",
            post(translations).layer(DefaultBodyLimit::max(MAX_AUDIO_BYTES)),
        )
}

// ---- audio -----------------------------------------------------------------------------------

async fn transcriptions(
    State(engine): State<AppState>,
    form: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    audio_to_text(engine, TranscriptionTask::Transcribe, form?).await
}

async fn translations(
    State(engine): State<AppState>,
    form: Result<Multipart, MultipartRejection>,
) -> Result<Response, ApiError> {
    audio_to_text(engine, TranscriptionTask::Translate, form?).await
}

async fn audio_to_text(
    engine: AppState,
    task: TranscriptionTask,
    form: Multipart,
) -> Result<Response, ApiError> {
    let (request, format) = parse_form(task, read_form(form).await?)?;
    let wanted_logprobs = request.logprobs;
    if request.stream {
        let stream = engine.transcribe(request).await?;
        // Transcript streams end with their done event, without chat's `[DONE]`.
        let events = stream.map(move |event| match event {
            Ok(TranscriptionEvent::Delta(d)) => sse_event(&TextDeltaEvent::new(d)),
            Ok(TranscriptionEvent::Done(t)) => sse_event(&TextDoneEvent::new(&t, wanted_logprobs)),
            Err(e) => sse_error(e),
        });
        return Ok(sse_only(events));
    }
    let transcript = transcription::collect(engine.transcribe(request.clone()).await?).await?;
    Ok(match render(&transcript, &request, format) {
        Rendered::Json(v) => Json(v).into_response(),
        Rendered::Text { mime, body } => ([(header::CONTENT_TYPE, mime)], body).into_response(),
    })
}

async fn speech_audio(
    State(engine): State<AppState>,
    body: Result<Json<CreateSpeechRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(req) = body?;
    let (request, stream_format) = req.into_core()?;
    let mime = request.format.mime();
    let audio = engine.speak(request).await?;
    if stream_format == StreamFormat::Sse {
        // Speech streams end with their done event, without chat's `[DONE]`.
        let events = audio.map(|chunk| match chunk {
            Ok(SpeechChunk::Audio(bytes)) => sse_event(&AudioDeltaEvent::new(&bytes)),
            Ok(SpeechChunk::Done(usage)) => sse_event(&AudioDoneEvent::new(usage)),
            Err(e) => sse_error(e),
        });
        return Ok(sse_only(events));
    }
    let (bytes, _) = speech::collect(audio).await?;
    Ok(([(header::CONTENT_TYPE, mime)], bytes).into_response())
}
