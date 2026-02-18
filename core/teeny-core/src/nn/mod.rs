/*
 * Copyright (c) 2026 Teenygrad.
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

pub mod activation;
pub mod embedding;
pub mod linear;
pub mod param;

pub mod module;
pub mod sequential;

#[cfg(feature = "training")]
pub mod loss;
#[cfg(feature = "training")]
pub mod optim;

// modules
// pub use embedding::Embedding;
pub use module::Module;
// pub use sequential::Sequential;

// activations
pub use activation::relu;
pub use activation::sigmoid;

// // losses
// pub use loss::bce_loss::BCELoss;

// // optimizers
// pub use optim::adam::Adam;
// pub use optim::adam::AdamBuilder;
