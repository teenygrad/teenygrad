# Teenygrad

> ⚠️ **Warning**: This project is still under active development and not yet ready for production use.

Teenygrad is a high performance Rust based ML training and inference library in the spirit of Pytorch and Tinygrad.

Our goals:

- Highly concurrent, memory-safe ML framework – Designed for scalability and safety.

- Low memory footprint – Optimized to run even on the smallest devices.

- No performance compromises – Hardware-accelerated wherever possible.

- Broad hardware support – Compatible with a wide variety of devices, not just NVIDIA and AMD.

- Extensible framework – Enables developers to build high-performance accelerators for their hardware.

- Embedded-friendly – Core training and inference components are no_std compatible.

- Full async support – Efficient asynchronous operations, even on embedded systems.

- Multi-threaded by default – Maximizes utilization of all available CPU cores.

## Community

[Join our Discord](https://discord.gg/Dvtasac8) to discuss the project or seek help with any issues.

## FAQ

### 1. Why create this project when PyTorch, TensorFlow, and Tinygrad already exist?

These frameworks excel for development and deployment on large-scale infrastructure. However, we believe the future of AI lies in devices of all sizes—from edge devices to massive clusters. Existing frameworks are relatively heavy; TensorFlow Lite comes closest to our vision, but we wanted a modern, memory-safe language instead of C.

Yet, we didn’t want to sacrifice distributed training or other advanced features offered by larger projects. Enter Teenygrad—a lightweight but powerful alternative.

### 2. You use MIT/Apache-licensed open-source projects but license your project under GPLv3. Isn’t that contradictory?

Our goal isn’t to simply repackage others’ work. We leverage existing projects as foundations where appropriate but aim to build something fundamentally new and unique.

We embrace open source because it fosters collaboration and learning—not to monetize startups or individuals. To avoid ambiguity (like the WordPress controversy), we’re clarifying upfront:

- Commercial use is allowed, but proprietary solutions require a commercial license.

- Startups and companies with under $3M annual revenue qualify for a free commercial license. Once the project is stable and ready for usage, we will provide a simple web interface for you to get a license.
