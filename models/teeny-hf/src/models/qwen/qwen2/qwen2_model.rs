/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use crate::models::{
    llama::llama_model::{
        LlamaAttention, LlamaDecoderLayer, LlamaForCausalLM, LlamaForQuestionAnswering,
        LlamaForSequenceClassification, LlamaForTokenClassification, LlamaMLP,
        LlamaPreTrainedModel,
    },
    mistral::MistralModel,
};

pub trait Qwen2RMSNorm {}

pub trait Qwen2MLP: LlamaMLP {}

pub trait Qwen2Attention: LlamaAttention {}

pub trait Qwen2DecoderLayer: LlamaDecoderLayer {}

pub trait Qwen2PreTrainedModel: LlamaPreTrainedModel {}

pub trait Qwen2Model: MistralModel {}

pub trait Qwen2ForCausalLM: LlamaForCausalLM {}

pub trait Qwen2ForSequenceClassification: LlamaForSequenceClassification {}

pub trait Qwen2ForTokenClassification: LlamaForTokenClassification {}

pub trait Qwen2ForQuestionAnswering: LlamaForQuestionAnswering {}
