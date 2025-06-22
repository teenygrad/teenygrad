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

use proc_macro::TokenStream;
use syn::{DeriveInput, parse_macro_input};

/// Example of how to create a derive macro using macro_rules!
/// This shows the pattern for creating a derive macro that can be used with #[derive(ModelConfig)]
#[proc_macro_derive(ModelConfig)]
pub fn model_config_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // Use the macro_rules! macro to generate the implementation
    let expanded = quote::quote! {
        impl ModelConfig for $name {
            fn architectures(&self) -> Vec<Architecture> {
                self.architectures.clone()
            }

            fn attention_bias(&self) -> bool {
                self.attention_bias
            }

            fn attention_dropout(&self) -> f32 {
                self.attention_dropout
            }

            fn bos_token_id(&self) -> usize {
                self.bos_token_id
            }

            fn eos_token_id(&self) -> usize {
                self.eos_token_id
            }

            fn head_dim(&self) -> usize {
                self.head_dim
            }

            fn hidden_act(&self) -> HiddenAct {
                self.hidden_act.clone()
            }

            fn hidden_size(&self) -> usize {
                self.hidden_size
            }

            fn initializer_range(&self) -> f32 {
                self.initializer_range
            }

            fn intermediate_size(&self) -> usize {
                self.intermediate_size
            }

            fn max_position_embeddings(&self) -> usize {
                self.max_position_embeddings
            }

            fn max_window_layers(&self) -> usize {
                self.max_window_layers
            }

            fn model_type(&self) -> ModelType {
                self.model_type.clone()
            }

            fn num_attention_heads(&self) -> usize {
                self.num_attention_heads
            }

            fn num_hidden_layers(&self) -> usize {
                self.num_hidden_layers
            }

            fn num_key_value_heads(&self) -> Option<usize> {
                self.num_key_value_heads
            }

            fn rms_norm_eps(&self) -> f32 {
                self.rms_norm_eps
            }

            fn rope_scaling(&self) -> Option<f32> {
                self.rope_scaling
            }

            fn rope_theta(&self) -> usize {
                self.rope_theta
            }

            fn sliding_window(&self) -> Option<f32> {
                self.sliding_window
            }

            fn tie_word_embeddings(&self) -> bool {
                self.tie_word_embeddings
            }

            fn torch_dtype(&self) -> TorchDtype {
                self.torch_dtype.clone()
            }

            fn transformers_version(&self) -> String {
                self.transformers_version.clone()
            }

            fn use_cache(&self) -> bool {
                self.use_cache
            }

            fn use_sliding_window(&self) -> bool {
                self.use_sliding_window
            }

            fn vocab_size(&self) -> usize {
                self.vocab_size
            }
        }
    };

    TokenStream::from(expanded)
}
