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

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub dependencies: Vec<Box<Node>>,
}

// /// Reference-counted pointer to a Value
// pub type ValueRef = Rc<RefCell<Value>>;

// impl Value {
//     /// Create a new value with autodifferentiation support
//     pub fn new(
//         data: Option<TensorData>,
//         operation: Box<dyn TensorOp>,
//         dependencies: Vec<ValueRef>,
//         requires_grad: bool,
//     ) -> Self {
//         let id = Uuid::new_v4().to_string();
//         let shape = data.as_ref().map(|d| d.shape().to_vec());
//         let grad = shape.map(ndarray::Array::zeros);

//         Value {
//             id,
//             data,
//             operation,
//             dependencies,
//             grad,
//             requires_grad,
//             retain_grad: false,
//         }
//     }

//     /// Accumulate gradient (for handling multiple paths in computation graph)
//     pub fn accumulate_grad(&mut self, grad: &TensorData) {
//         if let Some(g) = self.grad.as_mut() {
//             if Self::is_1d(grad.shape()) {
//                 let g1 = grad[0];
//                 *g += g1;
//             } else if Self::is_1d(g.shape()) {
//                 let g1 = g[0];
//                 *g = grad + g1;
//             } else {
//                 *g += grad;
//             }
//         } else {
//             self.grad = Some(grad.clone());
//         }
//     }

//     /// Clear the gradients of this value and all its dependencies
//     pub fn zero_grad(&mut self) {
//         self.grad = None;
//         self.dependencies
//             .iter()
//             .for_each(|v| v.borrow_mut().zero_grad());
//     }

//     /// Backward pass for this value
//     pub fn backward(&self) {
//         if self.requires_grad {
//             self.operation
//                 .backward(&self.dependencies, self.grad.as_ref().unwrap());
//         }
//     }

//     pub fn eval(&mut self) {
//         if self.operation.is_param() {
//             return;
//         }

//         self.data = Some(self.operation.eval(&self.dependencies));
//     }

//     pub fn is_1d(shape: &[usize]) -> bool {
//         shape.len() == 1
//     }
// }

// pub fn toposort_graph(value: &ValueRef) -> Vec<ValueRef> {
//     let mut sorted = Vec::new();
//     let mut visited = HashSet::<String>::new();

//     fn visit(value: &ValueRef, sorted: &mut Vec<ValueRef>, visited: &mut HashSet<String>) {
//         if visited.contains(&value.borrow().id) {
//             return;
//         }

//         visited.insert(value.borrow().id.clone());

//         for dependency in &value.borrow().dependencies {
//             visit(dependency, sorted, visited);
//         }

//         sorted.push(value.clone());
//     }

//     visit(value, &mut sorted, &mut visited);
//     sorted
// }
