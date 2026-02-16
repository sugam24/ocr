# LexiSight OCR Service

LexiSight is a high-performance microservice for Optical Character Recognition (OCR) and layout analysis, powered by the `rednote-hilab/dots.ocr` model (based on Qwen2-VL). It is containerized and designed to support both local development and high-throughput production environments.

## Features

*   **Advanced OCR & Layout Analysis**: Detects text, tables, figures, formulae, and headers with high accuracy using state-of-the-art vision-language models.
*   **Structured Output**: Returns a strictly formatted JSON response mapping every element to its bounding box, category, and textual content.
*   **Dual Engine Support**:
    *   **Hugging Face**: Standard implementation for development and consumer-grade hardware.
    *   **vLLM**: Optimized serving engine for production, offering significant throughput improvements on supported hardware.
*   **Production Ready**: Built with FastAPI, Docker, and standard observability practices.

## Deployment Configuration

The service behavior is controlled through environment variables defined in `.env`.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | LexiSight | Service identifier. |
| `MODEL_SOURCE` | `vllm` | **Core Setting**: Selects the inference backend. Use `huggingface` for development or `vllm` for production. |
| `MODEL_NAME` | `rednote-hilab/dots.ocr` | The specific model identifier from Hugging Face. |
| `MODEL_CACHE_DIR` | `Model` | Path where model weights are stored persistently. |
| `DEVICE` | `cuda` | Target device for inference (`cuda` or `cpu`). |
| `MAX_FILE_SIZE_MB` | `10` | Maximum allowed payload size for uploads. |

### Engine Selection Guide

**Hugging Face (`MODEL_SOURCE="huggingface"`)**
*   **Use Case**: Local development, lower VRAM availability (e.g., RTX 3060/4060).
*   **Characteristics**: Easier to set up on Windows/WSL; uses standard PyTorch eager execution. Slower but widely compatible.

**vLLM (`MODEL_SOURCE="vllm"`)**
*   **Use Case**: Production deployment on server-grade GPUs (A10g, A100, H100).
*   **Characteristics**: Uses PagedAttention and optimized kernels for maximum token throughput. Warning: Requires specific CUDA versions and strictly compliant hardware.

## Installation

### Prerequisites
*   Docker and Docker Compose
*   NVIDIA GPU with drivers and Container Toolkit installed (for GPU support).

### Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/sagea-ai/LexiSight
    cd LexiSight
    ```

2.  **Configure Environment**:
    Copy the example configuration and adjust as needed.
    ```bash
    cp .env.example .env
    ```
    *Editor's Note: If running on a laptop, set `MODEL_SOURCE="huggingface"` in your `.env` file first.*

3.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```
    The service will bind to port `8000`.

## API Reference

### POST /api/inference

Processes an image document and extracts layout and text information.

**Request**
*   **Content-Type**: `multipart/form-data`
*   **Body**: `file` (Binary image data: JPG, PNG, PDF)

**Response**
Returns a JSON object containing the full text and a list of detected blocks.

```json
{
  "text": "Full extracted text content...",
  "blocks": [
    {
      "bbox": [100, 50, 500, 100],
      "category": "Section-header",
      "text": "# Introduction"
    },
    {
      "bbox": [50, 400, 550, 600],
      "category": "Picture",
      "text": null
    }
  ],
  "model_version": "lexisight-v1"
}
```

## Performance & Optimization

### Shared GPU Hosting (Tesla T4 Example)
When hosting LexiSight alongside other services on a single GPU (like a standard 16GB Tesla T4), it is critical to tune the memory usage.

**The Math**:
*   **Model Size**: Qwen2-VL-2B (bfloat16) ≈ **4.5 GB**
*   **KV Cache Overhead**: ~1-2 GB
*   **Total Reserved**: ~6.4 GB
*   **T4 Capacity**: 16 GB

To safely run this model while leaving room for 4-5 other lightweight services, set the following in your `.env`:

```bash
# Reserve only ~40% of the GPU for LexiSight (approx 6.4GB)
VLLM_GPU_MEMORY_UTILIZATION=0.4
```
This leaves **~9.6 GB** of VRAM available for other parallel workloads.

### Tuning Parameters
*   **`VLLM_GPU_MEMORY_UTILIZATION`**: Controls the fraction of GPU memory vLLM blocks. (0.4 = 40%, 0.9 = 90%).
*   **`VLLM_MAX_MODEL_LEN`**: Reduces context window if you are extremely tight on memory. Default is 8192.

### GET /info

Returns diagnostic information about the current runtime configuration.

**Response**
```json
{
  "engine": "huggingface",
  "device": "cuda",
  "model": "rednote-hilab/dots.ocr",
  "api_version": "v1"
}
```

## Troubleshooting

**Performance Warning**
If you see logs mentioning "Flash attention not available" or "fallback to eager implementation", this is expected behavior when running in `huggingface` mode on hardware/drivers that do not support Flash Attention 2. The service remains fully functional but will operate at reduced speed.

**Memory Issues**
The default `vLLM` configuration aggressively reserves GPU memory properly. If you encounter Out of Memory (OOM) errors during startup, verify that no other processes are consuming VRAM, or switch to `huggingface` mode which allows for more granular memory management strategies.