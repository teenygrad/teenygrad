# Teenygrad

> ⚠️ **Warning**: This project is still under active development and not yet ready for production use.

Teenygrad is a high performance Rust based ML training and inference library in the spirit of Pytorch and Tinygrad.

Our goals:

- Highly concurrent, memory safe ML framework.
- Low memory footprint to support even the smallest devices.
- No compromises on performance, hardware accelerated wherever possible.
- Support a wide variety of devices not such NVidia and AMD.
- Provide the framework that anyone can use to build high performance   accelerators for their hardware.
- Embedded support built-in (core components of training and inference ar no-std safe).

## FAQ

### 1. Why did you create this project when pytorch, tensorflow and tinygrad already exist?

Those systems are great for development and deployment on massive infrastructure. However, we envision that future AI will exist on all sizes of devices. Each of these frameworks is fairly heavy, the closest to our vision is Tensorflow Lite. However, we would prefer to work with a modern memory safe language rather than C.

We didn't however want to give up on distributed training, and all the features that the bigger projects have. Hence, Teenygrad.

### 2. You are making use of other open source projects which are MIT/Apache licensed. But your project is GPL v3, isn't that a contradiction?

Our aim isn't to just re-package other projects. We use those projects where appropriate as foundation. But our aim is build something new and unique.

We love open source software, because it allows us to share and learn. It is isn't our aim to extort money from startups or even individuals to help learning and development.

In order to avoid the Wordpress debacle, we wish to make clear from day 1 that this project can be used in commercial software. But if you wish to build propietary solutions, then you should get a commercial license.

We intend to offer any startup or company with revenues less than $3 million per year a free commercial license for their proprietary solutions.
