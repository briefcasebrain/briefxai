# Multi-stage build for optimized image size
FROM rust:1.75 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifest files
COPY Cargo.toml Cargo.lock ./

# Build dependencies (this is cached if Cargo.toml hasn't changed)
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY src ./src
COPY briefxai_ui_data ./briefxai_ui_data
COPY openclio_ui_data ./openclio_ui_data
COPY clippy.toml rustfmt.toml ./

# Build the application
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1001 briefx

# Copy the binary from builder
COPY --from=builder /app/target/release/briefx /usr/local/bin/briefx

# Copy static assets
COPY --from=builder /app/briefxai_ui_data /app/briefxai_ui_data
COPY --from=builder /app/openclio_ui_data /app/openclio_ui_data

# Set ownership
RUN chown -R briefx:briefx /app

# Switch to non-root user
USER briefx

# Set working directory
WORKDIR /app

# Expose port (Cloud Run uses PORT env variable)
EXPOSE 8080

# Set environment variables for Cloud Run
ENV PORT=8080
ENV RUST_LOG=info

# Run the application
CMD ["briefx", "ui", "--port", "8080"]