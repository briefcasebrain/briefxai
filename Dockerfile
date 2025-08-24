FROM python:3.11-slim

WORKDIR /app

# Copy Python implementation
COPY python/ ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy UI files
COPY ui ./ui

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8080

# Expose port
EXPOSE 8080

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--timeout", "300", "app:app"]