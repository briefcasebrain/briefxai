# Simple Dockerfile for Cloud Run deployment
FROM rust:latest AS builder

WORKDIR /app

# Copy everything
COPY . .

# Build the application
RUN cargo build --release

# Runtime stage
FROM ubuntu:24.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /app/target/release/briefx /usr/local/bin/briefx

# Copy UI assets
COPY --from=builder /app/briefxai_ui_data /app/briefxai_ui_data

WORKDIR /app

EXPOSE 8080

ENV PORT=8080
ENV RUST_LOG=warn

CMD ["briefx", "ui", "--port", "8080"]