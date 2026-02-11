# PodCraft AI
End-to-end RAG (Retrieval-Augmented Generation) pipeline for automated podcast generation from newsletters, processing 50+ newsletters daily with optimized LLM inference and FastAPI deployment.
## Overview

PodCraft AI transforms newsletters into engaging podcasts using advanced RAG techniques, semantic chunking, prompt engineering, and optimized LLM inference. The system processes 50+ newsletters daily, generating 30-minute podcasts with 35% faster generation time and 40% runtime reduction through intelligent caching and batching.
## Key Features

- **End-to-End RAG Pipeline**: Semantic chunking, vector embeddings, and retrieval-augmented generation
- **50+ Newsletters Daily**: Automated processing pipeline handling multiple newsletter sources
- **35% Generation Time Reduction**: Optimized LLM inference with prompt tuning and batching for ~5,000 words
- **40% Runtime Reduction**: FastAPI, Redis caching, and ElevenLabs TTS integration
- **Semantic Chunking**: Advanced text segmentation preserving context and meaning
- **Prompt Engineering**: Optimized prompts for high-quality podcast generation
- **FastAPI Deployment**: High-performance REST API with async processing
- **Redis Caching**: Intelligent caching reducing redundant LLM calls
- **ElevenLabs TTS**: High-quality text-to-speech integration for podcast audio
## Project Structure

```
podcraft_ai/
├── src/
│   ├── rag/
│   │   ├── semantic_chunking.py    # Advanced text chunking
│   │   ├── vector_store.py          # Embedding and retrieval
│   │   └── retrieval.py              # RAG retrieval logic
│   ├── llm/
│   │   ├── prompt_engineering.py   # Optimized prompts
│   │   ├── inference.py             # Batched LLM inference
│   │   └── model_manager.py         # LLM model management
│   ├── api/
│   │   └── fastapi_server.py        # REST API server
│   └── audio/
│       └── tts_integration.py       # ElevenLabs TTS
├── config/
│   └── config.yaml
└── requirements.txt
```
## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Start API Server

```bash
python src/api/fastapi_server.py
```
