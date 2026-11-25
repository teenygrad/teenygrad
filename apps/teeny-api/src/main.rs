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

//! `teeny-api`: serve teenygrad LLM inference over an OpenAI-compatible HTTP API.

use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
use clap::Parser;
use teeny_api::{Engine, ModelEngine, ServedModel};
#[cfg(feature = "image")]
use teeny_llm::images::EchoImageModel;
#[cfg(feature = "llm")]
use teeny_llm::{
    chat::EchoChatModel,
    completion::EchoCompletionModel,
    embedding::EchoEmbeddingModel,
    models::{LoadOptions, LoadedModel, Model},
    moderation::EchoModerationModel,
};
#[cfg(feature = "audio")]
use teeny_llm::{speech::EchoSpeechModel, transcription::EchoTranscriptionModel};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: std::net::IpAddr,

    /// Port to listen on.
    #[arg(long, short, default_value_t = 8000)]
    port: u16,

    /// A model to serve, by tag (`family:size`, or the family for its default size).
    /// Repeat it to serve several. The choices are the models this build includes.
    #[cfg(feature = "llm")]
    #[arg(long = "model", short, ignore_case = true, value_parser = model_parser())]
    models: Vec<Model>,

    /// The weights folder, when serving one model. Defaults to the model's snapshot in
    /// the local Hugging Face cache.
    #[cfg(feature = "llm")]
    #[arg(long)]
    model_dir: Option<std::path::PathBuf>,

    /// The longest conversation, in tokens, each model keeps (its KV cache). Defaults to
    /// each model's own.
    #[cfg(feature = "llm")]
    #[arg(long)]
    context: Option<u32>,

    /// Also serve the echo models of the APIs this build includes, for trying them without
    /// weights: `teeny-echo` for chat, which echoes the last user message,
    /// `teeny-echo-complete` for text completions, which repeats the prompt,
    /// `teeny-echo-embed` for embeddings, `teeny-echo-moderate` for moderations, which
    /// flags the categories a text names as `flag:<category>` (with `llm`);
    /// `teeny-echo-transcribe` for transcriptions, which describes the recording, and
    /// `teeny-echo-speak` for speech, which beeps once a word (with `audio`);
    /// `teeny-echo-image` for images, which paints the prompt as a colour (with `image`).
    #[arg(long)]
    echo: bool,
}

#[cfg(feature = "llm")]
fn model_parser() -> impl clap::builder::TypedValueParser<Value = Model> {
    use clap::builder::{PossibleValuesParser, TypedValueParser};
    PossibleValuesParser::new(Model::tags()).try_map(|tag| tag.parse::<Model>())
}

/// Loads the models `args` names, each on a blocking thread (loading compiles kernels).
#[cfg(feature = "llm")]
async fn load_models(args: &Args) -> anyhow::Result<Vec<ServedModel>> {
    anyhow::ensure!(
        args.model_dir.is_none() || args.models.len() == 1,
        "--model-dir names one model's weights; give exactly one --model with it"
    );
    let mut loaded = Vec::new();
    for &model in &args.models {
        let mut options =
            LoadOptions::default().with_progress(move |what| tracing::info!(%model, "{what}"));
        if let Some(dir) = &args.model_dir {
            options = options.with_model_dir(dir.clone());
        }
        if let Some(n) = args.context {
            options = options.with_context_length(n);
        }
        tracing::info!(%model, "loading");
        let m = tokio::task::spawn_blocking(move || model.load(&options))
            .await
            .context("the loading thread panicked")?
            .with_context(|| format!("loading {model}"))?;
        tracing::info!(%model, parameters = ?m.parameters, "loaded");
        loaded.push(m.into());
    }
    Ok(loaded)
}

/// The echo models of every API this build includes.
fn echo_models() -> Vec<ServedModel> {
    #[allow(unused_mut)]
    let mut echo: Vec<ServedModel> = Vec::new();
    #[cfg(feature = "llm")]
    echo.extend(
        [
            LoadedModel::from_chat(Arc::new(EchoChatModel)),
            LoadedModel::from_completion(Arc::new(EchoCompletionModel)),
            LoadedModel::from_embedding(Arc::new(EchoEmbeddingModel)),
            LoadedModel::from_moderation(Arc::new(EchoModerationModel)),
        ]
        .map(ServedModel::from),
    );
    #[cfg(feature = "audio")]
    echo.extend([
        ServedModel::from_transcription(Arc::new(EchoTranscriptionModel)),
        ServedModel::from_speech(Arc::new(EchoSpeechModel)),
    ]);
    #[cfg(feature = "image")]
    echo.push(ServedModel::from_images(Arc::new(EchoImageModel)));
    echo
}

/// The models to serve: those `args` names, and the echo models with `--echo`.
async fn served_models(args: &Args) -> anyhow::Result<Vec<ServedModel>> {
    #[cfg(feature = "llm")]
    let mut served = load_models(args).await?;
    #[cfg(not(feature = "llm"))]
    let mut served = Vec::new();
    if args.echo {
        served.extend(echo_models());
    }
    if served.is_empty() {
        anyhow::bail!("nothing to serve: {}", nothing_to_serve());
    }
    Ok(served)
}

/// What to pass to serve something.
fn nothing_to_serve() -> String {
    #[cfg(feature = "llm")]
    {
        let tags = Model::tags();
        if tags.is_empty() {
            "pass --echo; this build includes no models (rebuild with a model feature, e.g. \
             --features functiongemma3,cuda)"
                .to_owned()
        } else {
            format!("pass --model <tag> (one of {}) or --echo", tags.join(", "))
        }
    }
    #[cfg(not(feature = "llm"))]
    "pass --echo".to_owned()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let addr = SocketAddr::new(args.host, args.port);

    let engine: Arc<dyn Engine> = Arc::new(ModelEngine::new(served_models(&args).await?)?);
    let models: Vec<String> = engine.models().into_iter().map(|m| m.id).collect();

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}; is another server using the port?"))?;
    tracing::info!(%addr, ?models, "teeny-api listening");

    axum::serve(listener, teeny_api::router(engine))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .context("server error")
}
