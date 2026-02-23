# Risk and Compliance DOC-AI

**Intelligent Medical Document Analysis System**

A local, privacy-first RAG (Retrieval-Augmented Generation) pipeline for question-answering over risk and compliance PDF documents. Upload a PDF, ask natural-language questions, and get sourced answers — all running on your own hardware via Ollama.

## UI Overview
<img width="1898" height="987" alt="image" src="https://github.com/user-attachments/assets/32b6e47e-b7de-4449-967c-fffa689fc817" />
---

## Features

- **PDF Ingestion** — Upload any PDF; pages are extracted, chunked, and embedded automatically.
- **Local LLM** — Powered by [Ollama](https://ollama.com/) with configurable models (default: `mistral:7b`, a cybeer-security LLM).
- **Semantic Search** — HuggingFace embeddings (`BAAI/bge-large-en-v1.5`) indexed in an in-memory ChromaDB vector store.
- **Source Attribution** — Every answer cites the page numbers and excerpts it was derived from.
- **Live Telemetry** — Real-time sidebar showing device (CUDA/CPU), GPU info, model config, and per-query pipeline timing (load, chunk, embed, generate).
- **Streaming Responses** — Typewriter-style word-by-word answer rendering.
- **Session Export** — Download the full Q&A session as a Markdown report with timestamps and source citations.
- **Dark / Light Mode** — Toggle between the Prometheus HUD (dark) and Clinical White (light) themes.
- **Smart Caching** — Vector stores are cached by content hash; re-asking questions on the same document skips re-embedding.
- **Lazy Model Loading** — Both the LLM and embedding model are initialized on first use (singleton pattern), not at import time.

---

## Architecture

```
main.py                         # Entry point
config/
  model.yaml                    # LLM + embedding model configuration
src/
  document/
    document_loader.py           # PDF loading via PyPDFLoader
    text_splitter.py             # Recursive character text splitting
  embedding/
    embed_model.py               # HuggingFace embedding (lazy singleton)
  llm/
    model.py                     # Ollama LLM client (lazy singleton)
  vectorstore/
    vectordb.py                  # In-memory ChromaDB with content-hash caching
  retriever/
    QA_chain.py                  # RAG pipeline: retriever + RetrievalQA chain
  ui/
    gradio_ui.py                 # Gradio web interface (Prometheus HUD theme)
```

### Data Flow

```
PDF Upload
    |
    v
document_loader  -->  text_splitter  -->  vector_database  -->  retriever
    (PyPDFLoader)      (RecursiveChar     (ChromaDB +            |
                        TextSplitter)      HuggingFace            v
                                           Embeddings)      RetrievalQA
                                                            (Ollama LLM)
                                                                |
                                                                v
                                                       Answer + Sources
                                                       + Timing Metrics
                                                                |
                                                                v
                                                         Gradio Chat UI
                                                    (streaming + telemetry)
```

---

## Prerequisites

| Dependency | Purpose |
|---|---|
| **Python 3.10+** | Runtime |
| **[Ollama](https://ollama.com/)** | Local LLM server |
| **CUDA GPU** (optional) | Accelerates embeddings and inference; falls back to CPU |
| **HuggingFace account** | Required for the `HF_TOKEN` to download embedding models |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/LLM-document-assistant.git
cd LLM-document-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token_here
```

Get your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 5. Install and start Ollama

```bash
# Install Ollama (see https://ollama.com/download)
# Then pull the medical model:
ollama pull meditron:7b
```

> **Other medical-domain models you can try:**
> - `CybersecurityRiskAnalyst` — Cyber security fine-tune of LLaMA 2
> - `llama3:8b` — General-purpose (strong at medical text)
>
> Change the model in `config/model.yaml` and pull it with `ollama pull <model>`.

### 6. Configure (optional)

Edit `config/model.yaml` to adjust the LLM and embedding settings:

```yaml
llm_config:
  model: "mistral:7b"        # Ollama model name
  temperature: 0.3            # Lower = more precise
  base_url: "http://127.0.0.1:11434"
  num_ctx: 4096               # Context window size
  num_gpu: 1                  # GPU layers to offload

embedding_model:
  model: "BAAI/bge-large-en-v1.5"
  cuda: True                  # Set to False for CPU-only
```

---

## Usage

### Run the application

```bash
python main.py
```

The Gradio UI will launch at **http://127.0.0.1:7860**.

### Workflow

1. **Upload** a PDF document using the sidebar file picker.
2. **Ask** a question in the text input (e.g., `"What are the top 3 differences between this 2024 update and the previous 2023 version regarding AML checks?"`).
3. **Read** the streamed answer with source citations (page numbers + excerpts).
4. **Review** the System Telemetry panel for pipeline timing and hardware info.
5. **Export** the session as a Markdown report via the "Export Report" button.
6. **Toggle** between dark and light mode with the theme button in the header.

---

## Project Structure Deep Dive

### `src/document/document_loader.py`
Loads PDFs via LangChain's `PyPDFLoader`. Validates file existence, extension, and non-empty content. Returns a list of `Document` objects (one per page).

### `src/document/text_splitter.py`
Splits documents into overlapping chunks (default: 1000 chars, 100 overlap) using `RecursiveCharacterTextSplitter`. Filters out empty/whitespace-only chunks.

### `src/embedding/embed_model.py`
Lazy-loaded singleton for `HuggingFaceEmbeddings`. Validates `HF_TOKEN`, reads the model name from config, and initializes once on first call.

### `src/llm/model.py`
Lazy-loaded singleton for `OllamaLLM`. Reads model name, URL, temperature, and context window from config. Connects to the local Ollama server.

### `src/vectorstore/vectordb.py`
Creates in-memory ChromaDB vector stores. Caches stores by a SHA-256 hash of chunk content, so the same document is never re-embedded twice in a session.

### `src/retriever/QA_chain.py`
Orchestrates the full RAG pipeline. Two entry points:
- `retriever_qa(file, query)` — Simple answer string.
- `retriever_qa_with_metadata(file, query)` — Returns `(answer, sources, metrics)` with per-step timing for the UI telemetry panel.

### `src/ui/gradio_ui.py`
The Gradio 5.x web interface. Features a custom dual-theme CSS system (dark/light), streaming chat, real-time telemetry sidebar, file upload, and session export.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ConnectionError: Could not connect to Ollama` | Make sure Ollama is running: `ollama serve` |
| `EnvironmentError: HF_TOKEN not found` | Add `HF_TOKEN=hf_...` to your `.env` file |
| `model not found` in Ollama | Run `ollama pull meditron:7b` (or your configured model) |
| Slow first query | Normal — the embedding model downloads on first use (~1.3 GB). Subsequent queries reuse the cached model. |
| `CUDA out of memory` | Set `cuda: False` in `config/model.yaml` to use CPU embeddings, or use a smaller model. |
| UI elements not rendering | Ensure `gradio>=5.0.0` is installed. Run `pip install --upgrade gradio`. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com/) (local inference) |
| RAG Framework | [LangChain](https://www.langchain.com/) |
| Embeddings | [HuggingFace sentence-transformers](https://huggingface.co/BAAI/bge-large-en-v1.5) |
| Vector Store | [ChromaDB](https://www.trychroma.com/) (in-memory) |
| PDF Parsing | [PyPDF](https://pypdf.readthedocs.io/) |
| Web UI | [Gradio 5.x](https://www.gradio.app/) |
| Config | YAML + python-dotenv |

---

## License

This project is for **research and educational purposes only**.
