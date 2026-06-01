# MojoRAG

High-Performance Local RAG System on Mojo + Python

MojoRAG is a fully local Retrieval-Augmented Generation (RAG) system. It uses **Mojo** for ultra-fast SIMD-optimized vector search and **Python** for orchestration, ML model integration, and API serving. All data stays on your device.

## Features

- **High Performance** — Mojo-powered vector search with SIMD optimization (AVX-512)
- **Fully Local** — No cloud services, complete data privacy
- **Smart Indexing** — Automatic text chunking with overlap and metadata extraction
- **Flexible Profiles** — Auto-detection of optimal settings based on available RAM
- **Markdown Support** — Parses `.md` and `.txt` files with YAML frontmatter
- **REST API** — FastAPI server with `/health`, `/index`, `/ask` endpoints
- **CLI Interface** — Rich-based interactive and non-interactive CLI
- **Docker Ready** — Full containerization with multi-stage builds
- **Strict Mode** — Answers only from provided context, no hallucinations

## Requirements

- **Docker** (recommended for all platforms including Windows)
- **RAM**: 16 GB minimum (balanced profile)
- **Disk**: ~5 GB free (models + index)

> **Windows users:** Mojo does not run natively on Windows. Use Docker or WSL2.

## Quick Start

### 1. Clone and build

```bash
git clone <repo-url>
cd mojorag
make build
```

### 2. Download models

```bash
python scripts/download_models.py --profile balanced
```

Available profiles: `low_memory`, `balanced`, `performance`, `all`.

### 3. Start the server

```bash
make run
```

The server starts at `http://localhost:8000`. First startup takes 30–60 seconds for model loading.

### 4. Index your documents

```bash
make index
```

Or via API:

```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"data_dir":"/app/data"}'
```

### 5. Ask a question

```bash
make ask
```

Or via API:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is MojoRAG?","k":3}'
```

## Usage

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health and index status |
| `POST` | `/index` | Index documents from a directory |
| `POST` | `/ask` | Ask a question, get an answer |

### CLI

```bash
# Interactive mode
docker exec -it mojorag python3 -m src.python.main ask

# Non-interactive mode
docker exec mojorag python3 -m src.python.main ask --question "What is MojoRAG?"

# System info
docker exec mojorag python3 -m src.python.main info

# Index documents
docker exec mojorag python3 -m src.python.main index --data-dir /app/data
```

## Configuration

### Hardware Profiles

| Profile | RAM | LLM Model | Context | Size |
|---------|-----|-----------|---------|------|
| `low_memory` | <16 GB | Phi-3-mini Q4_0 | 2048 | ~2.2 GB |
| `balanced` | 16–32 GB | Phi-3-mini Q4 | 4096 | ~2.5 GB |
| `performance` | 32+ GB | Llama-3-8B Q4_K_M | 8192 | ~5.5 GB |

Override via environment variable:

```bash
export MOJORAG_PROFILE=balanced
```

### Chunking Parameters

Edit `src/python/config.py`:

```python
CHUNK_SIZE = 512      # tokens per chunk
CHUNK_OVERLAP = 64    # overlap between chunks
MAX_CHUNKS_PER_QUERY = 5
```

## Docker

```bash
make build          # Build image (with cache)
make build-no-cache # Build image (no cache)
make run            # Start container
make stop           # Stop container
make logs           # View logs
make restart        # Restart container
make shell          # Open shell in container
make clean-docker   # Remove images and cache
```

## Testing

```bash
make test           # Run all tests
make test-e2e       # Run E2E tests only
```

## Project Structure

```
mojorag/
├── data/                    # User documents (mounted as volume)
├── index/                   # Binary index files
├── models/                  # Downloaded models (not in git)
├── prompts/                 # Jinja2 prompt templates
│   ├── answer.j2
│   └── answer_strict.j2
├── scripts/
│   └── download_models.py   # Model download script
├── src/
│   ├── mojo/               # Mojo source code (compiled to .so)
│   │   ├── bindings.mojo    # Python interop layer
│   │   ├── search.mojo      # Vector search core
│   │   └── chunker.mojo    # Text chunking
│   │       
│   └── python/             # Python source code
│       ├── __init__.py
│       ├── config.py        # Configuration and hardware profiles
│       ├── main.py          # CLI interface (Click + Rich)
│       ├── ingester.py      # Markdown parsing
│       ├── embedder.py      # Embedding generation (ONNX Runtime)
│       ├── retriever.py     # Search interface (Mojo + Python fallback)
│       ├── generator.py     # LLM inference (llama-cpp-python)
│       ├── orchestrator.py  # Central coordinator
│       └── server.py        # FastAPI server
├── tests/
│   ├── test_search.py       # Unit tests
│   └── test_e2e.py          # End-to-end tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .dockerignore
└── README.md
```

## Performance

### Algorithmic Complexity

| Component | Complexity | 10K chunks | 100K chunks |
|-----------|------------|------------|-------------|
| Vector search (brute-force) | O(N × dim) | ~15 ms | ~150 ms |
| Text chunking | O(L) | <1 ms | <10 ms |
| Embedding (batch) | O(B × L × d²) | ~200 ms | ~200 ms |
| LLM generation | O(T × L) | 5–15 sec | 5–15 sec |

> For collections over 50K chunks, consider HNSW index

## Troubleshooting

**"Индекс не загружен" / "Index not loaded"**
Run `POST /index` first. The index is stored in memory and must be rebuilt after container restart.

**"Mojo module not available, using Python fallback"**
The Mojo module requires `libKGENCompilerRTShared.so`. Rebuild with: `make build-no-cache`.

**"EOF when reading a line" in CLI**
Use `--question` flag for non-interactive mode:
```bash
docker exec mojorag python3 -m src.python.main ask --question "Your question"
```

**Out of Memory (OOM)**
Switch to `low_memory` profile: `export MOJORAG_PROFILE=low_memory`.

**SSL errors during build**
Add `--trusted-host` flags in Dockerfile or use a PyPI mirror.


## License

MIT License — see [LICENSE](LICENSE).
