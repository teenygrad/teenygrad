//! `teeny-llm-chat`: interactive command-line chat with a teenygrad LLM.

use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};

use anyhow::Context;
use clap::Parser;
use futures::StreamExt;
use teeny_llm_core::{
    ChatMessage, EchoEngine, Engine, FinishReason, GenerationRequest, Prompt, SamplingParams,
};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Model to chat with.
    #[arg(long, short, default_value = EchoEngine::MODEL_ID)]
    model: String,

    /// System prompt to start the conversation with.
    #[arg(long, short)]
    system: Option<String>,

    /// Maximum tokens per reply.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Sampling temperature.
    #[arg(long)]
    temperature: Option<f32>,
}

const HELP: &str = "commands: /clear  forget the conversation\n          /help   show this help\n          /exit   quit (or Ctrl-D)";

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let engine: Arc<dyn Engine> = Arc::new(EchoEngine::new());
    // Fail fast on a bad --model rather than on the first message.
    engine.model(&args.model)?;

    let initial: Vec<ChatMessage> = args.system.iter().map(ChatMessage::system).collect();
    let mut history = initial.clone();
    let sampling = SamplingParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        ..SamplingParams::default()
    };

    eprintln!(
        "teeny-llm-chat — model `{}` on engine `{}`. /help for commands.",
        args.model,
        engine.name()
    );

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut line = String::new();
    loop {
        print!("> ");
        stdout.flush()?;
        line.clear();
        if stdin
            .lock()
            .read_line(&mut line)
            .context("failed to read stdin")?
            == 0
        {
            println!();
            return Ok(());
        }
        let input = line.trim();
        match input {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/help" => {
                println!("{HELP}");
                continue;
            }
            "/clear" => {
                history.clone_from(&initial);
                println!("(conversation cleared)");
                continue;
            }
            cmd if cmd.starts_with('/') => {
                println!("unknown command `{cmd}`\n{HELP}");
                continue;
            }
            _ => {}
        }

        history.push(ChatMessage::user(input));
        let request = GenerationRequest {
            model: args.model.clone(),
            prompt: Prompt::Chat(history.clone()),
            sampling: sampling.clone(),
        };

        let mut reply = String::new();
        let result: anyhow::Result<Option<FinishReason>> = async {
            let mut tokens = engine.generate_stream(request)?;
            while let Some(chunk) = tokens.next().await {
                let chunk = chunk?;
                print!("{}", chunk.text);
                stdout.flush()?;
                reply.push_str(&chunk.text);
                if chunk.finish_reason.is_some() {
                    return Ok(chunk.finish_reason);
                }
            }
            Ok(None)
        }
        .await;
        println!();

        match result {
            Ok(finish_reason) => {
                if finish_reason == Some(FinishReason::Length) {
                    eprintln!("(reply truncated at --max-tokens)");
                }
                history.push(ChatMessage::assistant(reply));
            }
            Err(e) => {
                // Drop the unanswered user turn so a retry doesn't send it twice.
                history.pop();
                eprintln!("error: {e:#}");
            }
        }
    }
}
