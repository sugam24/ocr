# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition
COPY pyproject.toml .

# Install dependencies using uv (skip vllm to keep image light for HF mode)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
    fastapi "uvicorn[standard]" python-multipart \
    pydantic pydantic-settings pypdfium2 python-dotenv pillow \
    "transformers>=5.0.0" huggingface_hub accelerate

# Copy app code
COPY app /app/app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run command
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
