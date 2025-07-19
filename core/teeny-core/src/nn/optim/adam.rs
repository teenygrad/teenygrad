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

use std::marker::PhantomData;

use derive_builder::Builder;

use crate::{
    dtype::Dtype,
    graph::{NodeRef, zeros},
    nn::param::Param,
};

#[derive(Builder, Debug, Clone)]
pub struct Adam<N: Dtype, P: Param<N>> {
    #[builder(default = "0.001")]
    pub lr: f32,

    #[builder(default = "0.9")]
    pub beta1: f32,

    #[builder(default = "0.999")]
    pub beta2: f32,

    #[builder(default = "1e-8")]
    pub eps: f32,

    #[builder(default = "vec![]")]
    pub params: Vec<P>,

    // Internal state for Adam algorithm
    #[builder(default = "vec![]")]
    m: Vec<NodeRef<f32>>, // First moment (momentum)

    #[builder(default = "vec![]")]
    v: Vec<NodeRef<f32>>, // Second moment (velocity)

    #[builder(default = "0")]
    t: usize, // Time step

    _marker: PhantomData<N>,
}

impl<N: Dtype, P: Param<N>> Adam<N, P> {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            params: vec![],
            m: vec![],
            v: vec![],
            t: 0,
            _marker: PhantomData,
        }
    }

    pub fn params(&mut self, params: Vec<P>) {
        self.params = params;
        // Initialize momentum and velocity tensors for each parameter
        self.m.clear();
        self.v.clear();

        for param in &self.params {
            // Initialize with zeros, same shape as parameter
            self.m.push(zeros(param.shape()));
            self.v.push(zeros(param.shape()));
        }
    }

    pub fn zero_grad(&mut self) {
        todo!()
    }

    pub fn step(&mut self) {
        self.t += 1;

        for param in self.params.iter_mut() {
            // Get gradient for this parameter
            if let Some(_grad) = param.grad() {
                // let data = param.weights();
                // Get current parameter data

                // // Update biased first moment estimate: m = β₁ * m + (1 - β₁) * grad
                // self.m[i] = self.beta1 * &self.m[i] + (1.0 - self.beta1) * &grad;

                // // Update biased second moment estimate: v = β₂ * v + (1 - β₂) * grad²
                // let grad_squared = &grad * &grad;
                // self.v[i] = self.beta2 * &self.v[i] + (1.0 - self.beta2) * &grad_squared;

                // // Compute bias-corrected first moment: m̂ = m / (1 - β₁^t)
                // let m_hat = &self.m[i] / (1.0 - self.beta1.powi(self.t as i32));

                // // Compute bias-corrected second moment: v̂ = v / (1 - β₂^t)
                // let v_hat = &self.v[i] / (1.0 - self.beta2.powi(self.t as i32));

                // // Update parameter: θ = θ - α * m̂ / (√v̂ + ε)
                // let v_hat_sqrt = v_hat.mapv(|x| x.sqrt());
                // let denominator = &v_hat_sqrt + self.eps;
                // let update = &m_hat / &denominator;

                // // Apply update
                // param_data = param_data - self.lr * &update;

                // // Update the parameter tensor
                // param.value.borrow_mut().data = Some(param_data);
            }
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use crate::nn::loss::Loss;

//     use super::*;
//     use ndarray::array;

//     #[test]
//     fn test_adam_optimizer() {
//         // Create a simple optimization problem: minimize f(x, y) = x² + y²
//         // The minimum is at (0, 0)

//         // Initialize parameters
//         let x: Tensor = array![[2.0, 3.0], [4.0, 5.0]].into();
//         let y: Tensor = array![[1.0, 2.0], [3.0, 4.0]].into();

//         // Create optimizer
//         let mut optimizer = AdamBuilder::default().lr(0.05).build().unwrap();
//         optimizer.params(vec![x, y]);

//         // Optimization loop
//         for step in 0..100 {
//             // Zero gradients
//             optimizer.zero_grad();

//             // Forward pass: compute loss = x² + y²
//             let x_squared = &optimizer.params[0] * &optimizer.params[0];
//             let y_squared = &optimizer.params[1] * &optimizer.params[1];
//             let mut loss = Loss::new(&x_squared + &y_squared);

//             // Backward pass
//             loss.backward();

//             // Update parameters
//             optimizer.step();

//             // Print progress every 20 steps
//             if step % 20 == 0 {
//                 let x_val = optimizer.params[0].eval();
//                 let y_val = optimizer.params[1].eval();
//                 let loss_val = loss.loss.eval();
//                 println!("Step {step}: x={x_val:?}, y={y_val:?}, loss={loss_val:?}");
//             }
//         }

//         // Check that parameters have moved toward the minimum (0, 0)
//         let final_x = optimizer.params[0].eval();
//         let final_y = optimizer.params[1].eval();

//         // All values should be closer to 0 than the initial values
//         assert!(final_x.iter().all(|&v| v.abs() < 2.0));
//         assert!(final_y.iter().all(|&v| v.abs() < 1.5));
//     }

//     #[test]
//     fn test_adam_momentum() {
//         // Test that Adam properly tracks momentum across steps
//         let param: Tensor = array![[1.0]].into();

//         let mut optimizer = AdamBuilder::default().lr(0.1).build().unwrap();
//         optimizer.params(vec![param]);

//         // First step
//         optimizer.zero_grad();
//         let mut loss1 = Loss::new(&optimizer.params[0] * &optimizer.params[0]);
//         loss1.backward();

//         optimizer.step();

//         // Second step
//         optimizer.zero_grad();
//         let mut loss2 = Loss::new(&optimizer.params[0] * &optimizer.params[0]);
//         loss2.backward();
//         optimizer.step();

//         // Check that momentum state is maintained
//         assert_eq!(optimizer.t, 2);
//         assert_eq!(optimizer.m.len(), 1);
//         assert_eq!(optimizer.v.len(), 1);
//     }
// }
