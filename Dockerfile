# ----------- Build Stage -----------
FROM nvidia/cuda:12.3.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        pkg-config \
        libssl-dev \
        ca-certificates \
        cmake \
        python3 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set CUDA_HOME
ENV CUDA_HOME=/usr/local/cuda

# Set workdir and copy source
WORKDIR /workspace
COPY . .

# Build your project (replace 'your-binary-name' as needed)
RUN source $HOME/.cargo/env && cargo build --release

# ----------- Runtime Stage -----------
FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# Install runtime dependencies (add more if your binary needs them)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl-dev \
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*

# Set CUDA_HOME
ENV CUDA_HOME=/usr/local/cuda

# Set workdir
WORKDIR /app

# Copy the built binary from the builder stage
COPY --from=builder /workspace/target/release /app/release

# No default command specified; provide the command when running the container
