use clap::{Parser, ValueEnum};
use teeny_samples::ex02_simple_classifier;
use tracing::info;
use tracing_subscriber::{self, EnvFilter};

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    SimpleClassifier,
}

#[derive(Debug, Clone, ValueEnum)]
enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Parser)]
#[command(name = "teeny-samples")]
#[command(about = "A CLI application for running teenygrad models")]
struct Args {
    /// The model to run
    #[arg(
        value_enum,
        short = 'm',
        long = "model",
        default_value = "simple-classifier"
    )]
    model: Model,

    /// The log level
    #[arg(value_enum, short = 'l', long = "log-level", default_value = "info")]
    log_level: LogLevel,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let log_level = match args.log_level {
        LogLevel::Error => "error",
        LogLevel::Warn => "warn",
        LogLevel::Info => "info",
        LogLevel::Debug => "debug",
        LogLevel::Trace => "trace",
    };

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(log_level)),
        )
        .init();

    info!("Starting teeny-samples");

    teeny_runtime::init().unwrap();
    let devices = teeny_runtime::get_cuda_devices().unwrap();
    println!("Devices: {:?}", devices);

    match args.model {
        Model::SimpleClassifier => {
            info!("Running simple-classifier model");
            ex02_simple_classifier::run().await?;
        }
    }

    Ok(())
}
