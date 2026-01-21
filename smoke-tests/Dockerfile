# Base Image: PyTorch 2.9.1 with CUDA 12.6 support (verified stable for RTX 4070 Ti)
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /app

# 1. Install system tools (Git for potential pip installs from repos, curl for debugging)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy Smoke Test Script (assuming it's in src/ in your repo)
COPY src/train_smoke_test.py /app/src/

# 4. Environment Variables
# Force Python to print logs immediately (vital for K8s logs)
ENV PYTHONUNBUFFERED=1
# Set Hugging Face cache to a writable directory (can be shared via NFS)
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

# 5. Default Command
CMD ["python", "src/train_smoke_test.py"]