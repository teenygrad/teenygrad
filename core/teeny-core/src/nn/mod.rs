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

pub mod activation;
pub mod embedding;
pub mod linear;

pub mod macros;
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
pub use activation::relu::ReLU;
pub use activation::sigmoid::Sigmoid;

// // losses
// pub use loss::bce_loss::BCELoss;

// // optimizers
// pub use optim::adam::Adam;
// pub use optim::adam::AdamBuilder;
