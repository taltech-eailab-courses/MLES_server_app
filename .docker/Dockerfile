# Multi-architecture Dockerfile for ARM64 and AMD64
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CPU support (works for both ARM64 and AMD64)
# Using CPU-only version to reduce image size and ensure compatibility
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install llama-cpp-python with specific version (may require compilation)
# CMAKE_ARGS can be customized for different backends
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir llama-cpp-python==0.3.16

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    gradio \
    gradio_client \
    easyocr \
    pytesseract \
    transformers \
    accelerate \
    onnx \
    onnxruntime

# Copy application files
COPY models ./models
COPY services ./services
COPY app.py .

# Expose port for FastAPI
EXPOSE 8000

# Default command - using uvicorn for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]