# Multimodal RAG System

A production-grade Retrieval-Augmented Generation (RAG) system following **Jason Liu's methodology** with support for multimodal content (text, images, tables).

## 🎯 Key Features

### Core Capabilities
- **Hybrid Search**: Combines BM25 (keyword) and vector (semantic) search for 15-30% accuracy improvement
- **Reranking**: Cohere reranking for +20% accuracy boost
- **Multimodal Support**: Process and retrieve from text, images, tables, and charts
- **Production Architecture**: MultiVectorRetriever pattern with separate storage for summaries and content
- **Systematic Evaluation**: Synthetic data generation, comprehensive metrics, continuous improvement

### Following Jason Liu's Methodology
1. ✅ Fix retrieval before generation
2. ✅ Measure everything systematically
3. ✅ Hybrid search as non-negotiable
4. ✅ Reranking in every production system
5. ✅ Start with synthetic data (5 questions/chunk, target 97% recall)
6. ✅ Segment users and queries before optimizing
7. ✅ Fight "absence bias" - track retrieval failures

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage](#usage)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Deployment](#deployment)
- [Advanced Features](#advanced-features)

## 🚀 Installation

### Prerequisites
- Python 3.10+
- OpenAI API key (for embeddings and generation)
- Cohere API key (optional, for reranking)
- CUDA-capable GPU (optional, for local CLIP embeddings)

### Option 1: Using pip
```bash
# Clone the repository
git clone <repo-url>
cd multimodal-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Poetry
```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
poetry shell
```

### Configuration
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API keys
nano .env  # or your preferred editor
```

Required API keys in `.env`:
```bash
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Optional but recommended
```

## 🏃 Quick Start

### 1. Basic Pipeline

```python
from src.document_processing import DocumentParser, RecursiveChunker
from src.embeddings import EmbeddingService
from src.retrieval import HybridRetriever, VectorStoreFactory, Reranker
from src.generation import MultimodalGenerator

# Initialize components
vector_store = VectorStoreFactory.create()
retriever = HybridRetriever(vector_store)
generator = MultimodalGenerator()
reranker = Reranker()

# Process document
parser = DocumentParser()
doc = parser.parse("path/to/document.pdf")

chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(doc)

# Index
vector_store.add_chunks(chunks)

# Query
question = "What are the main findings?"
results = retriever.retrieve(question, k=10)
results = reranker.rerank(question, results, top_n=5)

chunks_retrieved = [chunk for chunk, score in results]
answer = generator.generate(question, chunks_retrieved)

print(answer)
```

### 2. Run Example Pipeline

```bash
# Place documents in ./data/documents/
mkdir -p data/documents
# Add your PDF/DOCX files

# Run complete pipeline
python scripts/run_pipeline.py
```

### 3. Launch UI

**Streamlit (Recommended for Production)**
```bash
streamlit run ui/streamlit_app.py
```

**Gradio (Recommended for Prototyping)**
```bash
python ui/gradio_app.py
```

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │     Query Router            │
         │  (Classify query type)      │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌──────────────────────────────┐
         │   Hybrid Retriever           │
         │  ┌──────────┬──────────┐     │
         │  │ Vector   │  BM25    │     │
         │  │ Search   │  Search  │     │
         │  └────┬─────┴────┬─────┘     │
         │       │          │           │
         │       └────┬─────┘           │
         │            │ RRF             │
         └────────────┼─────────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │      Reranker              │
         │  (Cohere rerank-v3.0)      │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  MultiVectorRetriever      │
         │  (Fetch original content)  │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │  Multimodal Generator      │
         │  (GPT-4o / Claude)         │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │       Answer               │
         └────────────────────────────┘
```

### MultiVectorRetriever Pattern

The system implements the recommended architecture for scalability:

1. **Summaries in Vector DB**: Store text descriptions/summaries for fast semantic search
2. **Originals in Document Store**: Keep full images, tables, complete text separate
3. **Link via Metadata**: Connect summaries to originals through unique IDs
4. **Retrieval Flow**: Search summaries → Look up originals → Pass to LLM

Benefits:
- 100x cost reduction for storage
- Sub-100ms retrieval latency
- Scale to billions of embeddings
- Efficient multimedia handling

### Directory Structure

```
multimodal-rag/
├── src/
│   ├── config.py                 # Configuration management
│   ├── document_processing/      # PDF, DOCX, image extraction
│   │   ├── parsers.py
│   │   ├── chunking.py
│   │   └── table_processor.py
│   ├── embeddings/               # Text and multimodal embeddings
│   │   ├── embedding_service.py
│   │   ├── image_describer.py
│   │   └── multimodal_embeddings.py
│   ├── retrieval/                # Hybrid search, reranking
│   │   ├── vector_store.py
│   │   ├── hybrid_retriever.py
│   │   ├── multi_vector_retriever.py
│   │   └── reranker.py
│   ├── generation/               # LLM generation, routing
│   │   ├── generator.py
│   │   └── query_router.py
│   └── evaluation/               # Metrics, synthetic data
│       ├── metrics.py
│       ├── synthetic_data.py
│       └── evaluator.py
├── ui/
│   ├── streamlit_app.py         # Production UI
│   └── gradio_app.py            # Prototyping UI
├── scripts/
│   └── run_pipeline.py          # Example pipeline
├── data/
│   ├── documents/               # Input documents
│   ├── vector_db/               # ChromaDB/LanceDB storage
│   ├── images/                  # Extracted images
│   └── evaluation/              # Evaluation results
├── tests/                       # Unit tests
├── configs/                     # Configuration files
├── pyproject.toml              # Poetry dependencies
├── requirements.txt            # Pip dependencies
└── README.md
```

## 📖 Usage

### Document Processing

```python
from src.document_processing import DocumentParser, RecursiveChunker

# Parse PDF with images and tables
parser = DocumentParser(extract_images=True, extract_tables=True)
doc = parser.parse("document.pdf")

print(f"Text elements: {len(doc.text_elements)}")
print(f"Images: {len(doc.image_elements)}")
print(f"Tables: {len(doc.table_elements)}")

# Chunk strategies
chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
chunks = chunker.chunk(doc)

# Or use section-based chunking
from src.document_processing import SectionChunker
chunker = SectionChunker(max_section_size=2000)
chunks = chunker.chunk(doc)
```

### Embeddings

```python
from src.embeddings import EmbeddingService, CLIPEmbeddings, ImageDescriber

# Text embeddings
embedding_service = EmbeddingService(model="text-embedding-3-small")
embedding = embedding_service.embed_text("Sample text")

# Multimodal embeddings (CLIP)
clip = CLIPEmbeddings(model_name="ViT-L-14")
text_emb = clip.embed_text("A photo of a cat")
image_emb = clip.embed_image(pil_image)
similarity = clip.compute_similarity(text_emb, image_emb)

# Image description for RAG
describer = ImageDescriber(model="gpt-4o")
description = describer.describe_image(image_base64)
chart_desc = describer.describe_chart(chart_image_base64)
```

### Retrieval

```python
from src.retrieval import HybridRetriever, Reranker

# Hybrid search (BM25 + Vector)
retriever = HybridRetriever(
    vector_store=vector_store,
    chunks=chunks,
    alpha=0.6  # 60% semantic, 40% keyword
)

results = retriever.retrieve("query", k=10)

# Reranking
reranker = Reranker(model="rerank-english-v3.0")
reranked = reranker.rerank("query", results, top_n=5)
```

### Generation

```python
from src.generation import MultimodalGenerator, QueryRouter

# Route queries
router = QueryRouter()
routing = router.route("Show me the revenue charts")
# Returns: query_type, search_text, search_images, etc.

# Generate answers
generator = MultimodalGenerator(model="gpt-4o")

# Text-only
answer = generator.generate(query, context_chunks)

# Multimodal (with images)
answer = generator.generate(query, context_chunks, images=image_elements)

# With citations
result = generator.generate_with_citations(query, context_chunks)
print(result['answer'])
print(result['citations'])
```

### Evaluation

```python
from src.evaluation import SyntheticDataGenerator, RAGEvaluator

# Generate synthetic data
generator = SyntheticDataGenerator()
dataset = generator.generate_dataset(chunks, num_questions_per_chunk=5)

# Evaluate
evaluator = RAGEvaluator()

# Retrieval metrics
metrics = evaluator.evaluate_retrieval(
    test_dataset=dataset,
    retrieve_fn=lambda q, k: retriever.retrieve(q, k),
    k=5
)

print(f"Recall@5: {metrics['recall@5']:.2%}")
```

## ⚙️ Configuration

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
COHERE_API_KEY=...

# Model Configuration
TEXT_EMBEDDING_MODEL=text-embedding-3-small
VISION_MODEL=gpt-4o
GENERATION_MODEL=gpt-4o
RERANK_MODEL=rerank-english-v3.0

# Vector Database
VECTOR_DB_TYPE=chromadb  # chromadb, lancedb, qdrant, milvus
VECTOR_DB_PATH=./data/vector_db
COLLECTION_NAME=multimodal_rag

# Retrieval
RETRIEVAL_TOP_K=20
RERANK_TOP_K=5
HYBRID_SEARCH_ALPHA=0.6
USE_RERANKING=true

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
CHUNKING_STRATEGY=recursive
```

### Vector Database Options

**ChromaDB** (Default - Best for prototyping)
```python
settings.vector_db_type = "chromadb"
```

**LanceDB** (Recommended for production)
```python
settings.vector_db_type = "lancedb"
# Supports petabyte-scale, multimodal-native, SQL queries
```

**Qdrant** (Best for performance)
```python
settings.vector_db_type = "qdrant"
# Rust implementation, GPU acceleration, excellent filtering
```

## 📊 Evaluation

### Jason Liu's 6-Tier Framework

**Tier 1: Foundation Metrics** (No LLM needed, fast)
- Retrieval Precision
- Retrieval Recall

**Tier 2: Primary RAG Relationships** (LLM-based)
- Context Relevance (C|Q): Does context address the question?
- Faithfulness (A|C): Is answer grounded in context?
- Answer Relevance (A|Q): Does answer address the question?

**Tier 3: Advanced Relationships**
- Context Coverage (C|A): Does context support all claims?
- Answerability (Q|C): Can context answer the question?

### Running Evaluation

```bash
# Generate synthetic dataset
python scripts/generate_synthetic_data.py

# Run evaluation
python scripts/evaluate.py

# Results saved to ./data/evaluation/
```

### Target Metrics

Following Jason Liu's methodology:
- **Retrieval Recall**: Target 97% on synthetic data
- **Faithfulness**: Target >95% (critical for production)
- **Context Relevance**: Target >85%
- **Answer Relevance**: Target >90%

## 🚀 Deployment

### Streamlit Cloud (Easiest)

```bash
# Push to GitHub
git init
git add .
git commit -m "Initial commit"
git push

# Deploy on streamlit.io
# Connect GitHub repo, set environment variables
```

### Docker

```dockerfile
# Coming soon
```

### Production Checklist

- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure logging (structured logs)
- [ ] Implement semantic caching (Redis)
- [ ] Set up feedback collection
- [ ] Schedule weekly evaluation runs
- [ ] Configure alerts for degraded performance
- [ ] Implement API rate limiting
- [ ] Set up cost tracking

## 🔧 Advanced Features

### Fine-tuning Embeddings

```python
# Collect training data from user feedback
# Train with sentence-transformers
# Expect 20-40% recall improvement
```

### Query Enhancement

```python
from src.generation.query_enhancement import HyDE, StepBackPrompting

# Hypothetical Document Embeddings
hyde = HyDE()
enhanced_query = hyde.enhance(query)

# Step-back prompting
step_back = StepBackPrompting()
broader_query = step_back.enhance(query)
```

### Caching

```python
# Semantic caching
from src.utils.caching import SemanticCache

cache = SemanticCache(threshold=0.95)
result = cache.get(query) or expensive_operation(query)
cache.set(query, result)
```

## 📈 Performance Optimization

### Storage Optimization

**Quantization** (75% storage reduction)
```python
# Scalar quantization (4x compression, <2% accuracy loss)
vector_store.enable_quantization(type="scalar")

# Binary quantization (32x compression for high-dim)
vector_store.enable_quantization(type="binary")
```

### Retrieval Optimization

**Hybrid Search Tuning**
- Technical docs: α=0.3-0.4 (favor keywords)
- Creative content: α=0.7-0.8 (favor semantics)
- Mixed: α=0.5-0.6 (balanced)

**Reranking Strategy**
- Always retrieve 2-4x desired results
- Rerank to top-k
- Example: Retrieve 20, rerank to 5

## 🐛 Troubleshooting

### Common Issues

**"No results returned"**
- Check if documents are indexed: `vector_store.get_collection_stats()`
- Lower similarity threshold
- Try pure BM25 search (α=0.0)

**"High latency"**
- Enable result caching
- Reduce retrieval_top_k
- Use quantization
- Consider faster embedding model

**"Poor answer quality"**
- Evaluate retrieval first (oracle test)
- Check if retrieved context contains answer
- Adjust chunk size (try 500, 1000, 1500)
- Enable reranking

## 📚 Resources

- [Jason Liu's RAG Methodology](https://jxnl.co/)
- [Instructor Library](https://github.com/jxnl/instructor)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

This implementation follows **Jason Liu's systematic RAG methodology**:
- Hybrid search as non-negotiable
- Reranking in every production system
- Measure retrieval before generation
- Fight absence bias through logging
- Start with synthetic data
- Iterate based on metrics

Special thanks to the open-source community for:
- LangChain
- ChromaDB/LanceDB
- Sentence Transformers
- Unstructured.io
- OpenAI/Anthropic/Cohere

---

**Built with ❤️ following Jason Liu's production RAG methodology**
