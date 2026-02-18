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

// use ndarray::s;
// use teeny_core::nn::loss::LossFn;
// use teeny_core::nn::{self, Module1};
// use teeny_core::sequential;
// use teeny_core::tensor::Tensor;
// use teeny_data::dataset::loader::load_csv;
// use tracing::info;

use teeny_core::nn;

pub struct SimpleClassifier<'data> {
    pub model: nn::sequential::Sequential<'data>,
}

// impl<'a> Default for SimpleClassifier<'a> {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl<'a> nn::Module1<&Tensor, Tensor> for SimpleClassifier<'a> {
//     fn forward(&self, _x: &Tensor) -> Tensor {
//         // self.model.forward(x)
//         todo!()
//     }

//     fn parameters(&self) -> Vec<Tensor> {
//         self.model.parameters()
//     }
// }

// impl<'a> SimpleClassifier<'a> {
//     pub fn new() -> Self {
//         Self {
//             model: sequential![
//                 nn::Linear::new(8, 12, false),
//                 nn::ReLU::new(),
//                 nn::Linear::new(12, 8, false),
//                 nn::ReLU::new(),
//                 nn::Linear::new(8, 1, false),
//                 nn::Sigmoid::new()
//             ],
//         }
//     }
// }

// pub async fn run() -> Result<(), Box<dyn std::error::Error>> {
//     let dataset = load_csv::<f32>(
//         "https://raw.githubusercontent.com/teenygrad/data/main/pima-indians-diabetes/diabetes.csv",
//         b',',
//     )
//     .await
//     .unwrap();

//     let x = dataset.slice(s![.., ..8]);

//     let y = dataset.slice(s![.., 8]);
//     let y = y.to_shape((dataset.shape()[0], 1)).unwrap();

//     let model = SimpleClassifier::new();
//     let mut optimizer = nn::AdamBuilder::default()
//         .params(model.parameters())
//         .build()
//         .unwrap();
//     let loss_fn = nn::BCELoss::new();
//     const BATCH_SIZE: usize = 10;

//     for epoch in 0..100 {
//         for i in (0..y.shape()[0]).step_by(BATCH_SIZE) {
//             let range_start = i;
//             let range_end = std::cmp::min(i + BATCH_SIZE, y.shape()[0]);
//             let x_batch = x.slice(s![range_start..range_end, ..]).into();
//             let y_pred = model.forward(&x_batch);

//             let y_batch = y.slice(s![range_start..range_end, ..]).into();
//             let mut loss = loss_fn.compute(&y_pred, &y_batch);

//             optimizer.zero_grad();
//             loss.backward();
//             info!("Loss: {:?}", loss.loss.value.borrow().data);

//             optimizer.step();
//         }

//         info!("Epoch {:?}, loss {:?}", epoch, epoch);
//     }

//     Ok(())
// }

// # load the dataset, split into input (X) and output (y) variables
// dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
// X = dataset[:,0:8]
// y = dataset[:,8]

// X = torch.tensor(X, dtype=torch.float32)
// y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

// # define the model
// model = nn.Sequential(
//     nn.Linear(8, 12),
//     nn.ReLU(),
//     nn.Linear(12, 8),
//     nn.ReLU(),
//     nn.Linear(8, 1),
//     nn.Sigmoid()
// )
// print(model)

// # train the model
// loss_fn   = nn.BCELoss()  # binary cross entropy
// optimizer = optim.Adam(model.parameters(), lr=0.001)

// n_epochs = 100
// batch_size = 10

// for epoch in range(n_epochs):
//     for i in range(0, len(X), batch_size):
//         Xbatch = X[i:i+batch_size]
//         y_pred = model(Xbatch)
//         ybatch = y[i:i+batch_size]
//         loss = loss_fn(y_pred, ybatch)
//         optimizer.zero_grad()
//         loss.backward()
//         optimizer.step()
//     print(f'Finished epoch {epoch}, latest loss {loss}')

// # compute accuracy (no_grad is optional)
// with torch.no_grad():
//     y_pred = model(X)
// accuracy = (y_pred.round() == y).float().mean()
// print(f"Accuracy {accuracy}")
