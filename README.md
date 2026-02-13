# LightOnOCR Service

A high-performance OCR microservice powered by [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) — a state-of-the-art 1B-parameter vision-language model for converting documents (PDFs, scans, images) into clean, naturally ordered text.

## Features

*   **State-of-the-Art OCR**: Powered by LightOnOCR-2-1B, achieving top performance on OlmOCR-Bench while being ~9× smaller and significantly faster than competing approaches.
*   **End-to-End**: Fully differentiable, no brittle OCR pipeline — the model directly converts images to text.
*   **Versatile**: Handles tables, receipts, forms, multi-column layouts, math notation, and more.
*   **Dual Engine Support**:
    *   **Hugging Face**: Standard implementation for development and consumer-grade hardware.
    *   **vLLM**: Optimized serving engine for production with high throughput (~5.71 pages/s on a single H100).
*   **Production Ready**: Built with FastAPI, Docker, and `uv` for dependency management.

## Deployment Configuration

The service behavior is controlled through environment variables defined in `.env`.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | LightOnOCR-Service | Service identifier. |
| `MODEL_SOURCE` | `huggingface` | Selects the inference backend. Use `huggingface` for local dev or `vllm` for production. |
| `MODEL_NAME` | `lightonai/LightOnOCR-2-1B` | The model identifier from Hugging Face. |
| `MODEL_CACHE_DIR` | `Model` | Path where model weights are stored persistently. |
| `DEVICE` | `cuda` | Target device for inference (`cuda`, `cpu`, or `mps`). |
| `MAX_FILE_SIZE_MB` | `10` | Maximum allowed payload size for uploads. |

### Engine Selection Guide

**Hugging Face (`MODEL_SOURCE="huggingface"`)**
*   **Use Case**: Local development, lower VRAM availability.
*   **Requirements**: `transformers>=5.0.0`, GPU with ~4GB VRAM (bfloat16) or CPU fallback.
*   **Characteristics**: Easier to set up; uses standard PyTorch inference.

**vLLM (`MODEL_SOURCE="vllm"`)**
*   **Use Case**: Production deployment on server-grade GPUs (A10g, A100, H100).
*   **Setup**: Requires starting a separate vLLM server:
    ```bash
    vllm serve lightonai/LightOnOCR-2-1B \
        --limit-mm-per-prompt '{"image": 1}' \
        --mm-processor-cache-gb 0 \
        --no-enable-prefix-caching
    ```
*   **Characteristics**: Uses PagedAttention and optimized kernels for maximum throughput.

## Installation

### Local Development (with `uv`)

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd LightOnOCR-Service
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Configure environment**:
    ```bash
    cp .env.example .env
    ```

4.  **Run the server**:
    ```bash
    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The model will be downloaded automatically on first startup (~2GB).

### Docker Deployment

1.  **Configure environment**:
    ```bash
    cp .env.example .env
    ```

2.  **Build and run**:
    ```bash
    docker-compose up --build
    ```
    The service will bind to port `8000`.

## API Reference

### POST /api/inference

Processes an image or PDF document and extracts text using OCR.

**Request**
*   **Content-Type**: `multipart/form-data`
*   **Body**: `file` (Binary image data: JPG, PNG, PDF)

**Response**
```json
{
  "text": "Extracted text content from the document...",
  "blocks": [],
  "model_version": "lightonocr-2-1b"
}
```

### GET /info

Returns diagnostic information about the current runtime configuration.

**Response**
```json
{
  "engine": "huggingface",
  "device": "cuda",
  "model": "lightonai/LightOnOCR-2-1B",
  "api_version": "v1"
}
```

### GET /health

Health check endpoint.

**Response**
```json
{
  "status": "ok"
}
```

## Quick Test

```bash
# Upload an image for OCR
curl -X POST http://localhost:8000/api/inference \
  -F "file=@/path/to/your/document.png"

# Check service info
curl http://localhost:8000/info
```

## Performance

*   **Speed**: 3.3× faster than Chandra OCR, 5× faster than dots.ocr, 2× faster than PaddleOCR
*   **Efficiency**: ~493k pages/day on a single H100 for <$0.01 per 1,000 pages
*   **Model Size**: ~2GB (bfloat16), fits comfortably on consumer GPUs

### PDF Preprocessing

PDFs are rendered at 200 DPI (scale factor ≈ 2.77) as recommended by the model authors. Aspect ratio is maintained to preserve text geometry.

## Troubleshooting

**Model Download Issues**
If the model fails to download, ensure you have internet access and sufficient disk space (~2GB). You can also manually download the model:
```bash
huggingface-cli download lightonai/LightOnOCR-2-1B --local-dir Model
```

**Memory Issues**
LightOnOCR-2-1B is a 1B parameter model requiring ~2-4GB VRAM in bfloat16. If you encounter OOM errors:
- Switch to CPU: Set `DEVICE="cpu"` in `.env`
- Ensure no other processes are consuming VRAM

**transformers Version**
LightOnOCR-2-1B requires `transformers>=5.0.0`. If you get import errors for `LightOnOcrForConditionalGeneration`, upgrade:
```bash
uv pip install --upgrade transformers
```

## License

This service uses the [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) model, licensed under Apache License 2.0.

## Citation

```bibtex
@misc{lightonocr2_2026,
  title = {LightOnOCR: A 1B End-to-End Multilingual Vision-Language Model for State-of-the-Art OCR},
  author = {Said Taghadouini and Adrien Cavaillès and Baptiste Aubertin},
  year = {2026},
  howpublished = {\url{https://arxiv.org/abs/2601.14251}}
}
```