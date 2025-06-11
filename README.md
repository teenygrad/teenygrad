# Teenygrad

> ⚠️ **Warning**: This project is still under active development and not yet ready for production use.

## Community

[Join our Discord](https://discord.gg/Dvtasac8) to discuss the project or seek help with any issues.

## Teenygrad Vision

**Empowering the next generation of AI everywhere—uncompromised performance, unparalleled accessibility.**

We envision a world where machine learning is truly ubiquitous: from the tiniest embedded sensors to the largest distributed clusters, every device can harness the power of AI efficiently, safely, and without constraints.

By redefining lightweight, hardware-agnostic ML frameworks, we break down the barriers between cutting-edge research and real-world deployment. Teenygrad isn’t just a library—it’s the foundation for a future where:

Safety meets scalability: Memory-safe, concurrent, and extensible by design, enabling developers to push boundaries without sacrificing reliability.

Every device is an AI device: From microcontrollers to data centers, performance is never limited by hardware or legacy software bloat.

Innovation is democratized: Open collaboration fuels advancements, while our licensing ensures fairness for both creators and enterprises.

We’re not just building tools for today’s ML—we’re architecting the ecosystem for tomorrow’s intelligent edge.

## Mission

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

## Roadmap

- v0.1.0/Jun 30 - Compile Rust based kernel to PTX (nVidia 3090)

- v0.2.0/Jul 31 - Run Llama model on nVidia 3090

- v0.3.0/Sep 30 - Compile Triton kernels to SPIR-V

- v0.4.0/Oct 31 - Run Llama model on Rockchip 3588 using Vulkan/SPIR-V

- v0.5.0/Dec 31 - Run Deepseek R1 on Moorethreads MT4000

- 2026 and beyond
  - Q1: Training/Pytorch JIT
  - Q2: Teeny LLM server
  - Q3: Performance optimization
  - Q4: Sparsity/Quantization support
  - Q1: Observability/metrics
  - Q2: Model storage/versioning

## FAQ

### 1. Why create this project when PyTorch, TensorFlow, and Tinygrad already exist?

These frameworks excel for development and deployment on large-scale infrastructure. However, we believe the future of AI lies in devices of all sizes—from edge devices to massive clusters. Existing frameworks are relatively heavy; TensorFlow Lite comes closest to our vision, but we wanted a modern, memory-safe language instead of C.

Yet, we didn’t want to sacrifice distributed training or other advanced features offered by larger projects. Enter Teenygrad—a lightweight but powerful alternative.

### 3. I am interested in contributing to the project, how do I get started?

We welcome contributions both from companies and individuals, please see the CONTRIBUTIONG.md file for more information.

### 2. You use MIT/Apache-licensed open-source projects but license your project under GPLv3. Isn’t that contradictory?

Our goal isn’t to simply repackage others’ work. We leverage existing projects as foundations where appropriate but aim to build something fundamentally new and unique.

We embrace open source because it fosters collaboration and learning—not to monetize startups or individuals. To avoid ambiguity (like the WordPress controversy), we’re clarifying upfront:

- Commercial use is allowed, but proprietary solutions require a commercial license.

- Startups and companies with under $3M annual revenue qualify for a free commercial license. Once the project is stable and ready for usage, we will provide a simple web interface for you to get a license.

- Free commercial license holders get a basic level of support, companies on higher tiers get additional support options as well as an SLA.
