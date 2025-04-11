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
use teeny_tensor::tensor::Tensor;

mod mnist;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Directory to cache downloaded MNIST data
    #[arg(short, long, default_value = "/tmp/mnist")]
    cache_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let (mut x_train, mut y_train) = mnist::read_mnist_train_data(&args.cache_dir).await?;
    let (_x_test, _y_test) = mnist::read_mnist_test_data(&args.cache_dir).await?;

    let _x_train = x_train.reshape(Vec::from([-1, 28 * 28]));
    let _y_train = y_train.reshape(Vec::from([-1, 28 * 28]));

    println!("Hello, world!");
    Ok(())
}
