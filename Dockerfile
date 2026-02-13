# Use vLLM base image for pre-compiled CUDA/Torch/vLLM deps
FROM vllm/vllm-openai:latest

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition
COPY pyproject.toml .

# Install dependencies using uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system .

# Copy app code
COPY app /app/app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Reset entrypoint from base image
ENTRYPOINT []

# Run command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
