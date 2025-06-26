use clap::{Parser, ValueEnum};
use teeny_samples::simple_classifier;

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    SimpleClassifier,
}

#[derive(Parser)]
#[command(name = "teeny-samples")]
#[command(about = "A CLI application for running teenygrad models")]
struct Args {
    /// The model to run
    #[arg(value_enum, short = 'm', long = "model")]
    model: Model,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.model {
        Model::SimpleClassifier => {
            println!("Running simple-classifier model");
            simple_classifier::run().await?;
        }
    }

    Ok(())
}
