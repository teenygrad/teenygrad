/*
 * Copyright (c) SpinorML 2025. All rights reserved.
 *
 * This software and associated documentation files (the "Software") are proprietary
 * and confidential. The Software is protected by copyright laws and international
 * copyright treaties, as well as other intellectual property laws and treaties.
 *
 * No part of this Software may be reproduced, distributed, or transmitted in any
 * form or by any means, including photocopying, recording, or other electronic or
 * mechanical methods, without the prior written permission of SpinorML.
 *
 * Unauthorized copying, modification, distribution, or use of this Software is
 * strictly prohibited and may result in severe civil and criminal penalties.
 */

use anyhow::Result;
use clap::Parser;

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
    let (_x_train, _y_train) = mnist::fetch_mnist_train_data(&args.cache_dir).await?;
    let (_x_test, _y_test) = mnist::fetch_mnist_test_data(&args.cache_dir).await?;

    println!("Hello, world!");
    Ok(())
}
