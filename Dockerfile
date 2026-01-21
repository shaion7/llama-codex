# Base Image: PyTorch 2.9.1 with CUDA 12.6 support (Verified User Version)
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# 1. Install system tools
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy Production Training Script
# Assumes your local folder structure is llama-codex/src/train.py
COPY src/train.py /app/src/

# 4. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# 5. Default Command
CMD ["python", "src/train.py"]