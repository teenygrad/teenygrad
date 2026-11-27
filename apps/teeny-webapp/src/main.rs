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

//! `teeny-webapp`: serve the teenygrad browser UI, talking to a teeny-api server.

use std::net::SocketAddr;

use anyhow::Context;
use clap::Parser;
use teeny_webapp::{Config, app};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Address to bind.
    #[arg(long, default_value = "127.0.0.1")]
    host: std::net::IpAddr,

    /// Port to listen on.
    #[arg(long, short, default_value_t = 8080)]
    port: u16,

    /// The teeny-api server the UI talks to, as the browser reaches it.
    #[arg(long, default_value = "http://127.0.0.1:8000")]
    api_url: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let api_url = args.api_url.trim_end_matches('/').to_owned();
    anyhow::ensure!(
        api_url.starts_with("http://") || api_url.starts_with("https://"),
        "--api-url must be an http:// or https:// URL, not `{api_url}`"
    );
    let addr = SocketAddr::new(args.host, args.port);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("failed to bind {addr}; is another server using the port?"))?;
    tracing::info!(%api_url, "teeny-webapp listening on http://{addr}");

    axum::serve(listener, app(Config { api_url }))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("shutting down");
        })
        .await
        .context("server error")
}
