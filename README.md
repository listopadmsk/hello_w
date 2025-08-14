## RAG App (FastAPI + FAISS/NumPy)

Minimal Retrieval-Augmented Generation (RAG) service with:
- Offline fallback embeddings (hash) and stub LLM (no external keys required)
- Optional OpenAI or Ollama providers
- Vector index persisted on disk (FAISS if installed, otherwise NumPy)
- REST API for ingest and query

### Quickstart

1) Create a virtual environment and install deps
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# optionally, for faster search:
# pip install faiss-cpu==1.8.0.post1
```

2) Configure (optional)
```bash
cp .env.example .env
# edit .env as needed (defaults: EMBEDDING_PROVIDER=hash, LLM_PROVIDER=stub)
```

3) Add or use sample data
- Place your `.txt`/`.md`/`.pdf` files under `data/`
- A sample file is provided in `data/sample.txt`

4) Build the index
```bash
python scripts/ingest.py data
```

5) Run the API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

6) Query
```bash
curl -sS -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "Что такое RAG?", "top_k": 4}' | jq
```

### Providers
- Embeddings: `hash` (default, offline), `openai`
- LLM: `stub` (default, offline), `openai`, `ollama`

Set providers via `.env`:
```
EMBEDDING_PROVIDER=hash
LLM_PROVIDER=stub
OPENAI_API_KEY=...            # if using OpenAI
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:latest
```

### Scripts
- `python scripts/ingest.py data` — rebuild vector index from a directory
- `python scripts/ask.py "Ваш вопрос"` — ask a question using the current index

### Notes
- PDFs require `pypdf`; if missing, PDFs will be skipped
- Index is persisted under `storage/` by default; configure via `STORAGE_DIR`