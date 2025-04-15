/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use anyhow::Result;
use clap::Parser;
use smol::block_on;
use teeny_data::mnist;
use teeny_tensor::tensor::Tensor;

mod model;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Directory to cache downloaded MNIST data
    #[arg(short, long, default_value = "/tmp/mnist")]
    cache_dir: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    block_on(async {
        let (mut x_train, mut y_train) = mnist::read_mnist_train_data(&args.cache_dir).await?;
        let (_x_test, _y_test) = mnist::read_mnist_test_data(&args.cache_dir).await?;

        let _x_train = x_train.reshape(Vec::from([-1, 28 * 28]));
        let _y_train = y_train.reshape(Vec::from([-1, 28 * 28]));

        println!("Hello, world!");
        Ok(())
    })
}

// def train(model, X_train, Y_train, optim, steps, BS=128, lossfn=lambda out,y: out.sparse_categorical_crossentropy(y),
//         transform=lambda x: x, target_transform=lambda x: x, noloss=False):
//   with Tensor.train():
//     losses, accuracies = [], []
//     for i in (t := trange(steps, disable=CI)):
//       samp = np.random.randint(0, X_train.shape[0], size=(BS))
//       x = Tensor(transform(X_train[samp]), requires_grad=False)
//       y = Tensor(target_transform(Y_train[samp]))

//       # network
//       out = model.forward(x) if hasattr(model, 'forward') else model(x)

//       loss = lossfn(out, y)
//       optim.zero_grad()
//       loss.backward()
//       if noloss: del loss
//       optim.step()

//       # printing
//       if not noloss:
//         cat = out.argmax(axis=-1)
//         accuracy = (cat == y).mean().numpy()

//         loss = loss.detach().numpy()
//         losses.append(loss)
//         accuracies.append(accuracy)
//         t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
//   return [losses, accuracies]

// def evaluate(model, X_test, Y_test, num_classes=None, BS=128, return_predict=False, transform=lambda x: x,
//              target_transform=lambda y: y):
//   Tensor.training = False
//   def numpy_eval(Y_test, num_classes):
//     Y_test_preds_out = np.zeros(list(Y_test.shape)+[num_classes])
//     for i in trange((len(Y_test)-1)//BS+1, disable=CI):
//       x = Tensor(transform(X_test[i*BS:(i+1)*BS]))
//       out = model.forward(x) if hasattr(model, 'forward') else model(x)
//       Y_test_preds_out[i*BS:(i+1)*BS] = out.numpy()
//     Y_test_preds = np.argmax(Y_test_preds_out, axis=-1)
//     Y_test = target_transform(Y_test)
//     return (Y_test == Y_test_preds).mean(), Y_test_preds

//   if num_classes is None: num_classes = Y_test.max().astype(int)+1
//   acc, Y_test_pred = numpy_eval(Y_test, num_classes)
//   print("test set accuracy is %f" % acc)
//   return (acc, Y_test_pred) if return_predict else acc
