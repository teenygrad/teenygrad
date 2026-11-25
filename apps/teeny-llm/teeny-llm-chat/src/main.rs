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

//! `teeny-llm-chat`: interactive command-line chat with a teenygrad LLM.

use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
    sync::Arc,
};

use anyhow::Context;
use clap::Parser;
use futures::StreamExt;
use serde_json::Value;
use teeny_llm_core::chat::{
    AssistantMessage, ChatEvent, ChatFinishReason, ChatRequest, EchoChatModel, Message, Tool,
    ToolCall, ToolCallKind, ToolKind,
};
use teeny_llm_core::models::{LoadOptions, LoadedModel, Model};
use teeny_llm_core::{Engine, ModelEngine};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// The model to chat with, by tag (`family:size`, or the family for its default
    /// size); the choices are the models this build includes. Without it, `teeny-echo`,
    /// which echoes what you type.
    #[arg(long, short, ignore_case = true, value_parser = model_parser())]
    model: Option<Model>,

    /// The model's weights folder. Defaults to its snapshot in the local Hugging Face
    /// cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// The longest conversation, in tokens, the model keeps. Defaults to the model's own.
    #[arg(long)]
    context: Option<u32>,

    /// A JSON file of tool declarations, OpenAI style
    /// (`[{"type": "function", "function": {"name": ..., "parameters": ...}}]`). The
    /// model's calls are shown, and you type each result.
    #[arg(long)]
    tools: Option<PathBuf>,

    /// System prompt to start the conversation with.
    #[arg(long, short)]
    system: Option<String>,

    /// Maximum tokens per reply.
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Sampling temperature, for models that sample.
    #[arg(long)]
    temperature: Option<f32>,
}

fn model_parser() -> impl clap::builder::TypedValueParser<Value = Model> {
    use clap::builder::{PossibleValuesParser, TypedValueParser};
    PossibleValuesParser::new(Model::tags()).try_map(|tag| tag.parse::<Model>())
}

const HELP: &str = "commands: /clear  forget the conversation\n          /help   show this help\n          /exit   quit (or Ctrl-D)";

/// The function declarations in an OpenAI-style `tools` array.
fn read_tools(path: &PathBuf) -> anyhow::Result<Vec<Tool>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading the tools file {}", path.display()))?;
    let json: Value =
        serde_json::from_str(&text).with_context(|| format!("{} is not JSON", path.display()))?;
    let items = json
        .as_array()
        .with_context(|| format!("{} must hold a JSON array of tools", path.display()))?;
    items
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let f = t
                .get("function")
                .filter(|_| t["type"] == "function")
                .with_context(|| format!("tools[{i}] is not a function tool"))?;
            Ok(Tool {
                name: f["name"]
                    .as_str()
                    .with_context(|| format!("tools[{i}].function.name must be a string"))?
                    .to_owned(),
                description: f["description"].as_str().map(str::to_owned),
                kind: ToolKind::Function {
                    parameters: f.get("parameters").filter(|p| !p.is_null()).cloned(),
                    strict: false,
                },
            })
        })
        .collect()
}

fn load(args: &Args) -> anyhow::Result<LoadedModel> {
    let Some(model) = args.model else {
        return Ok(LoadedModel::from_chat(Arc::new(EchoChatModel)));
    };
    let mut options = LoadOptions::default().with_progress(|what| eprintln!("  [{what}]"));
    if let Some(dir) = &args.model_dir {
        options = options.with_model_dir(dir.clone());
    }
    if let Some(n) = args.context {
        options = options.with_context_length(n);
    }
    eprintln!("loading {model}...");
    model
        .load(&options)
        .with_context(|| format!("loading {model}"))
}

/// Reads a line, or `None` at end of input.
fn read_line(prompt: &str) -> anyhow::Result<Option<String>> {
    print!("{prompt}");
    io::stdout().flush()?;
    let mut line = String::new();
    if io::stdin()
        .lock()
        .read_line(&mut line)
        .context("failed to read stdin")?
        == 0
    {
        println!();
        return Ok(None);
    }
    Ok(Some(line.trim().to_owned()))
}

/// Streams one reply to the terminal, returning it as an assistant message and why it
/// ended.
async fn reply(
    engine: &dyn Engine,
    request: ChatRequest,
) -> anyhow::Result<(AssistantMessage, ChatFinishReason)> {
    let mut events = engine.chat_stream(request)?;
    let mut message = AssistantMessage::default();
    let mut finish = ChatFinishReason::Stop;
    let mut stdout = io::stdout();
    while let Some(event) = events.next().await {
        match event? {
            ChatEvent::Delta { index: 0, delta } => {
                if let Some(text) = delta.content {
                    print!("{text}");
                    stdout.flush()?;
                    message.content.get_or_insert_default().push_str(&text);
                }
                for d in delta.tool_calls {
                    let i = d.index as usize;
                    if message.tool_calls.len() <= i {
                        message.tool_calls.resize_with(i + 1, || ToolCall {
                            id: String::new(),
                            name: String::new(),
                            kind: ToolCallKind::Function,
                            input: String::new(),
                        });
                    }
                    let call = &mut message.tool_calls[i];
                    if let Some(id) = d.id {
                        call.id = id;
                    }
                    if let Some(name) = d.name {
                        call.name = name;
                    }
                    call.input.push_str(&d.input);
                }
            }
            ChatEvent::Finish { index: 0, reason } => finish = reason,
            _ => {}
        }
    }
    if message.content.is_some() {
        println!();
    }
    Ok((message, finish))
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let tools = args
        .tools
        .as_ref()
        .map(read_tools)
        .transpose()?
        .unwrap_or_default();
    let loaded = load(&args)?;
    let model = loaded.info.id.clone();
    let engine: Arc<dyn Engine> = Arc::new(ModelEngine::new(vec![loaded])?);

    let initial: Vec<Message> = args.system.iter().map(Message::system).collect();
    let mut history = initial.clone();
    eprintln!(
        "teeny-llm-chat: {model}{}. /help for commands.",
        if tools.is_empty() {
            String::new()
        } else {
            format!(
                ", tools: {}",
                tools
                    .iter()
                    .map(|t| t.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    );

    loop {
        let Some(input) = read_line("> ")? else {
            return Ok(());
        };
        match input.as_str() {
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

        let turn_start = history.len();
        history.push(Message::user(input));
        // A turn can take several replies: each round of tool calls is answered, then the
        // model goes on.
        let result: anyhow::Result<()> = async {
            loop {
                let request = ChatRequest {
                    model: model.clone(),
                    messages: history.clone(),
                    tools: tools.clone(),
                    max_completion_tokens: args.max_tokens,
                    sampling: teeny_llm_core::chat::Sampling {
                        temperature: args.temperature,
                        ..Default::default()
                    },
                    ..ChatRequest::default()
                };
                let (message, finish) = reply(engine.as_ref(), request).await?;
                let calls = message.tool_calls.clone();
                history.push(Message::Assistant(message));
                if finish == ChatFinishReason::Length {
                    eprintln!("(reply truncated at --max-tokens)");
                }
                if calls.is_empty() {
                    return Ok(());
                }
                for call in calls {
                    println!("  call {}: {}({})", call.id, call.name, call.input);
                    let Some(result) = read_line(&format!("  {} result> ", call.name))? else {
                        return Ok(());
                    };
                    history.push(Message::Tool {
                        tool_call_id: call.id,
                        content: result,
                    });
                }
            }
        }
        .await;
        if let Err(e) = result {
            // Drop the unanswered turn so a retry doesn't send it twice.
            history.truncate(turn_start);
            eprintln!("error: {e:#}");
        }
    }
}
