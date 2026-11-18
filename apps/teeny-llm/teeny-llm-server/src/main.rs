//! `teeny-llm-server`: serve teenygrad LLM inference over an OpenAI-compatible HTTP API.

use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
use clap::Parser;
use teeny_llm_core::{EchoEngine, Engine};
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

    let engine: Arc<dyn Engine> = Arc::new(EchoEngine::new());
    let models: Vec<String> = engine.models().into_iter().map(|m| m.id).collect();

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}; is another server using the port?"))?;
    tracing::info!(%addr, engine = engine.name(), ?models, "teeny-llm-server listening");

    axum::serve(listener, teeny_llm_server::router(engine))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .context("server error")
}
