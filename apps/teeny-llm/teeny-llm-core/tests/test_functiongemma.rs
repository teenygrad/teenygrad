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

//! FunctionGemma end to end against PyTorch (FP64): the prompt as one step, then
//! greedy decoding one token per step over the KV cache. Every step's logits are
//! compared, and the greedy tokens must match exactly.
//!
//! Needs the model (`GEMMA_MODEL_DIR`, the Hugging Face snapshot folder) and fixtures
//! from `tests/fixtures/functiongemma/gen_reference.py` (`GEMMA_FIXTURES`); ignored by default.

use std::path::PathBuf;

use teeny_llm_core::models::functiongemma::GemmaConfig;
use teeny_llm_core::models::functiongemma::runner::{CompiledStep, Session, Weights, argmax};
use teeny_test::model::runtime;

fn ids(path: &PathBuf) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect()
}

fn f32s(path: &PathBuf) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Max |got - want| over the logits, and over their range for scale.
fn diff(got: &[f32], want: &[f32]) -> (f32, f32) {
    assert_eq!(got.len(), want.len(), "logits length");
    let mut max = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.is_finite(), "non-finite logit at {i}");
        max = max.max((g - w).abs());
    }
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    (max, scale)
}

/// The prompt `gen_reference.py` renders, built in Rust with the model's own chat
/// template and tokenizer, must give the same token ids.
#[test]
#[ignore = "needs GEMMA_MODEL_DIR and GEMMA_FIXTURES (gated model, fixtures not committed)"]
fn test_chat_template_matches_transformers() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var("GEMMA_MODEL_DIR").expect("set GEMMA_MODEL_DIR"));
    let fix = PathBuf::from(std::env::var("GEMMA_FIXTURES").expect("set GEMMA_FIXTURES"));
    let chat = teeny_llm_core::models::functiongemma::chat::Chat::load(&model_dir)?;
    let messages =
        serde_json::json!([{"role": "user", "content": "What is the weather in Paris?"}]);
    let tools = serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]);
    let got = chat.prompt(&messages, &tools)?;
    let want = ids(&fix.join("prompt_ids.txt"));
    if got != want {
        println!("rendered: {:?}", chat.render(&messages, &tools)?);
    }
    assert_eq!(got, want, "prompt token ids differ from transformers'");

    let text = chat.decode(&ids(&fix.join("generated_ids.txt")))?;
    let calls = teeny_llm_core::models::functiongemma::chat::parse_calls(&text)?;
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].args, serde_json::json!({"city": "Paris"}));
    Ok(())
}

#[test]
#[ignore = "needs GEMMA_MODEL_DIR and GEMMA_FIXTURES (gated model, fixtures not committed)"]
fn test_functiongemma_matches_pytorch() -> anyhow::Result<()> {
    let model_dir = PathBuf::from(std::env::var("GEMMA_MODEL_DIR").expect("set GEMMA_MODEL_DIR"));
    let fix = PathBuf::from(std::env::var("GEMMA_FIXTURES").expect("set GEMMA_FIXTURES"));
    let steps: usize = std::env::var("GEMMA_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);

    let cfg = GemmaConfig::load(&model_dir.join("config.json"))?;
    let weights = Weights::load(&model_dir.join("model.safetensors"))?;
    let prompt = ids(&fix.join("prompt_ids.txt"));
    let want_ids = ids(&fix.join("generated_ids.txt"));
    let steps = steps.min(want_ids.len());

    let device = runtime::open()?;
    let capacity = (prompt.len() + want_ids.len()).next_multiple_of(64);
    let mut session = Session::new(cfg.clone(), capacity);
    let mut prefill = CompiledStep::new(
        &device,
        &cfg,
        &weights,
        prompt.len(),
        capacity,
        &session.past,
    )?;
    let mut decode = CompiledStep::new(&device, &cfg, &weights, 1, capacity, &session.past)?;

    let mut worst = 0.0f32;
    let mut logits = session.step(&device, &mut prefill, &prompt)?;
    for (i, &want_id) in want_ids.iter().enumerate().take(steps) {
        let want = f32s(&fix.join(format!("logits_{i}.bin")));
        let (d, scale) = diff(&logits, &want);
        let tok = argmax(&logits);
        println!(
            "step {i}: max |{} - torch| = {d:e} (max |logit| {scale:.2}), token {tok} (torch {})",
            runtime::BACKEND_NAME,
            want_id
        );
        worst = worst.max(d);
        assert_eq!(tok, want_id, "step {i}: greedy token differs");
        assert!(d < 1e-2, "step {i}: logits off by {d:e}");
        if i + 1 < steps {
            logits = session.step(&device, &mut decode, &[tok])?;
        }
    }
    println!("worst logit difference over {steps} steps: {worst:e}");
    Ok(())
}

/// FunctionGemma through the factory and the chat API: loaded with `Model::load`, served
/// by `ModelEngine`, asked about the weather with one tool offered, it must call that tool
/// for the city. A second turn with the tool's result must reuse the cached prompt.
#[tokio::test]
#[ignore = "needs GEMMA_MODEL_DIR (gated model)"]
async fn test_functiongemma_serves_chat_with_tool_calls() -> anyhow::Result<()> {
    use std::sync::Arc;

    use teeny_llm_core::chat::{
        AssistantMessage, ChatFinishReason, ChatRequest, Message, Tool, ToolKind, collect,
    };
    use teeny_llm_core::models::{LoadOptions, Model};
    use teeny_llm_core::{Engine, ModelEngine};

    let model_dir = PathBuf::from(std::env::var("GEMMA_MODEL_DIR").expect("set GEMMA_MODEL_DIR"));
    let options = LoadOptions::default()
        .with_model_dir(model_dir)
        .with_context_length(512)
        .with_progress(|s| eprintln!("  [{s}]"));
    let loaded =
        tokio::task::spawn_blocking(move || Model::FunctionGemma3.load(&options)).await??;
    assert!(loaded.parameters.is_some_and(|p| p > 200_000_000));
    let tokenizer = loaded
        .tokenizer
        .clone()
        .expect("FunctionGemma has a tokenizer");
    assert!(tokenizer.count("What is the weather in Paris?")? > 0);
    let engine: Arc<dyn Engine> = Arc::new(ModelEngine::new(vec![loaded])?);

    let weather = Tool {
        name: "get_weather".into(),
        description: Some("Get current weather".into()),
        kind: ToolKind::Function {
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            })),
            strict: false,
        },
    };
    let mut request = ChatRequest {
        model: "functiongemma3".into(),
        messages: vec![Message::user("What is the weather in Paris?")],
        tools: vec![weather],
        max_completion_tokens: Some(64),
        include_usage: true,
        ..ChatRequest::default()
    };
    let first = collect(
        Model::FunctionGemma3.tag(),
        1,
        engine.chat_stream(request.clone())?,
    )
    .await?;
    let choice = &first.choices[0];
    assert_eq!(
        choice.finish_reason,
        ChatFinishReason::ToolCalls,
        "{choice:?}"
    );
    let call = &choice.message.tool_calls[0];
    assert_eq!(call.name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&call.input)?,
        serde_json::json!({"city": "Paris"})
    );

    request.messages.push(Message::Assistant(AssistantMessage {
        tool_calls: choice.message.tool_calls.clone(),
        ..AssistantMessage::default()
    }));
    request.messages.push(Message::Tool {
        tool_call_id: call.id.clone(),
        content: r#"{"temperature_c": 18, "conditions": "sunny"}"#.into(),
    });
    let second = collect(Model::FunctionGemma3.tag(), 1, engine.chat_stream(request)?).await?;
    println!("second turn: {:?}", second.choices[0].message.content);
    assert!(
        second.usage.cached_prompt_tokens >= first.usage.prompt_tokens / 2,
        "the second turn should reuse the first's KV cache: {:?}",
        second.usage
    );
    Ok(())
}
