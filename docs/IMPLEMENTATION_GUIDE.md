# Multimodal RAG System Implementation Plan

## Overview
Comprehensive RAG system with multimodal ingestion (web, PDF, video), AI agents, and MCP servers.

## Architecture

### Backend (Python + LangChain)
- **RAG Pipeline**: Ingest → Chunk → Embed → Store → Retrieve → Generate
- **Vector DB**: Pinecone (scalable, hybrid metadata filters)
- **LLMs**: GPT-4-turbo (primary), Claude 3.5 Sonnet (backup)
- **Embeddings**: text-embedding-3-large (3072 dims)
- **Multimodal**: CLIP ViT-L/14 for video frames, Whisper for transcription

### MCP Servers (Modular Tool Access)
1. **File Server** (port 5001): Controlled filesystem access
2. **Web Fetch Server** (port 5002): Rate-limited web scraping with allowlist
3. **Pinecone Server** (port 5003): Vector DB operations
4. **Transcription Server** (port 5004): Whisper audio/video transcription
5. **Vision Server** (port 5005): Frame extraction + CLIP embeddings

### AI Agents
1. **Ingestion Agent**: Orchestrates document ingestion pipeline
2. **Retrieval Agent**: Dual retriever (text + frames) with reranking
3. **Chat Agent**: LLM generation with streaming and citations

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment with uv
cd backend
uv venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Copy and configure environment
cp ../.env.example ../.env
# Edit .env with your API keys
```

### 2. Configure API Keys

Update `.env` with your actual keys:
- `OPENAI_API_KEY`: From https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY`: From https://console.anthropic.com/settings/keys
- `PINECONE_API_KEY`: From https://app.pinecone.io/
- `LANGCHAIN_API_KEY`: From https://smith.langchain.com/settings

### 3. Setup Pinecone Index

```bash
# Create Pinecone index
python scripts/setup_pinecone.py
```

### 4. Start MCP Servers

```bash
# Terminal 1: File Server
python ../mcp-servers/file-server/file_mcp.py

# Terminal 2: Web Fetch Server
python ../mcp-servers/web-fetch-server/web_fetch_mcp.py

# Terminal 3: Transcription Server
python ../mcp-servers/transcription-server/transcription_mcp.py

# Terminal 4: Vision Server
python ../mcp-servers/vision-server/vision_mcp.py
```

### 5. Start Backend API

```bash
# Terminal 5: Main API
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start Frontend (Next.js)

```bash
cd ../frontend
npm install
npm run dev
```

## Usage Examples

### Ingest URL
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "source_type": "url",
    "namespace": "default"
  }'
```

### Ingest PDF
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "/path/to/document.pdf",
    "source_type": "pdf",
    "namespace": "default"
  }'
```

### Ingest Video
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "uri": "/path/to/video.mp4",
    "source_type": "video",
    "namespace": "default"
  }'
```

### Query with Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "top_k": 8,
    "namespace": "default"
  }'
```

## Data Flow

### Ingestion Pipeline
1. **Source Detection**: URL/PDF/Video
2. **Content Extraction**:
   - URL: MCP Web Fetch → HTML → Markdown
   - PDF: PyPDF → Text extraction
   - Video: FFmpeg frames + Whisper transcription
3. **Chunking**: RecursiveCharacterTextSplitter (800 tokens, 150 overlap)
4. **Embedding**: 
   - Text → OpenAI text-embedding-3-large
   - Frames → CLIP ViT-L/14
5. **Storage**: Pinecone with rich metadata (source, page, timestamp, modality)

### Retrieval Pipeline
1. **Query Embedding**: text-embedding-3-large
2. **Vector Search**: Pinecone similarity search (top-k × 2)
3. **Reranking**: Cross-encoder (ms-marco-MiniLM-L-6-v2)
4. **Context Assembly**: Format with citations [Source N | Page X | t=MM:SS]

### Generation Pipeline
1. **Prompt Construction**: System guardrails + context + query
2. **LLM Call**: GPT-4-turbo streaming
3. **Citation Extraction**: Parse and format sources
4. **Response**: Streaming answer with source attributions

## Metadata Schema

```json
{
  "source": "url|pdf|video",
  "uri": "file:///path or https://url",
  "modality": "text|frame",
  "page": 3,
  "frame_sec": 120.5,
  "chunk_id": "hash_page_chunk",
  "hash": "sha256_content",
  "timestamp": "2026-01-09T12:00:00Z",
  "mime": "application/pdf|text/html|video/mp4",
  "title": "Document Title"
}
```

## Monitoring

### LangSmith Tracing
- Enable: `LANGCHAIN_TRACING_V2=true`
- View traces: https://smith.langchain.com/
- Tracks: Token usage, latency, retrieval quality

### Health Checks
```bash
# Backend
curl http://localhost:8000/health

# MCP Servers
curl http://localhost:5001/health  # File
curl http://localhost:5002/health  # Web Fetch
curl http://localhost:5004/health  # Transcription
curl http://localhost:5005/health  # Vision
```

## Advanced Features

### Multimodal Search
- Query text retrieves relevant text chunks AND video frames
- Timestamps in citations enable video playback navigation
- CLIP embeddings allow visual similarity search

### Reranking
- Cross-encoder improves precision
- Configurable via `use_reranking=true` in query

### Namespace Management
- Separate collections per dataset
- Filter by namespace in queries

### Rate Limiting
- MCP Web Fetch: 10 req/min per domain
- Configurable domain allowlist

## Next Steps

1. **Implement Frontend UI** (Next.js chat interface)
2. **Add Evaluation Framework** (golden QA sets, RAGAS metrics)
3. **Implement Monitoring Agent** (cost tracking, drift detection)
4. **Add Document Deduplication** (semantic + hash-based)
5. **Scale with Kubernetes** (containerize MCP servers)
6. **Add Advanced Video Features** (scene detection, OCR on frames)

## Troubleshooting

### Pinecone Connection Issues
- Verify API key and environment in `.env`
- Check index exists: `python scripts/setup_pinecone.py`

### MCP Server Not Responding
- Ensure correct port in `.env`
- Check server logs for errors
- Verify network connectivity

### Video Processing Fails
- Install ffmpeg: `apt-get install ffmpeg` (Linux) or `brew install ffmpeg` (Mac)
- Check video codec compatibility
- Reduce FFMPEG_MAX_FRAMES if memory constrained

### Embedding Dimensions Mismatch
- Ensure OPENAI_EMBEDDING_DIMS=3072 in `.env`
- Recreate Pinecone index if changed

## Performance Tuning

- **Chunking**: Adjust CHUNK_SIZE/OVERLAP for your docs
- **Retrieval**: Increase TOP_K for better recall, adjust with reranking
- **Video**: Balance FRAME_INTERVAL vs. coverage
- **Costs**: Use sentence-transformers for embeddings if budget-constrained

## Security Considerations

- API keys in `.env` (never commit!)
- MCP domain allowlisting for web scraping
- File server path restrictions
- Rate limiting on all MCP servers
