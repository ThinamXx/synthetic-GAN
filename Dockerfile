# Use Python 3.9 slim image as base.
FROM python:3.9-slim

# Set environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory.
WORKDIR /app

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching.
COPY requirements.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files.
COPY . .

# Install the package.
RUN pip install -e .

# Create non-root user.
RUN useradd --create-home --shell /bin/bash syngan && \
    chown -R syngan:syngan /app
USER syngan

# Create necessary directories.
RUN mkdir -p data/{raw,processed,synthetic} models outputs

# Expose port for potential web interface.
EXPOSE 8000

# Default command.
CMD ["python", "src/scripts/main.py"]