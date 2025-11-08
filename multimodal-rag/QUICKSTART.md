# Quick Start Guide

Get up and running with Multimodal RAG in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Clone and setup
git clone <repo-url>
cd multimodal-rag

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY and COHERE_API_KEY
```

## 2. Prepare Documents (1 minute)

```bash
# Create data directory
mkdir -p data/documents

# Add your PDF or DOCX files
cp /path/to/your/documents/*.pdf data/documents/
```

## 3. Run Pipeline (2 minutes)

```bash
# Process documents and build index
python scripts/run_pipeline.py
```

This will:
- ✓ Parse documents (extract text, images, tables)
- ✓ Chunk content (1000 tokens, 200 overlap)
- ✓ Generate embeddings
- ✓ Build hybrid search index
- ✓ Generate synthetic evaluation data
- ✓ Run evaluation metrics

## 4. Launch UI

### Option A: Streamlit (Production)

```bash
streamlit run ui/streamlit_app.py
```

Open browser to http://localhost:8501

### Option B: Gradio (Prototyping)

```bash
python ui/gradio_app.py
```

Open browser to http://localhost:7860

## 5. Try It Out!

### Upload a Document

1. Go to "Documents" tab
2. Upload PDF/DOCX files
3. Click "Process Documents"
4. Wait for indexing to complete

### Ask Questions

1. Go to "Query" tab
2. Type your question: "What are the main findings?"
3. Click "Search"
4. View answer and retrieved context

## Next Steps

### Optimize Performance

**Tune Hybrid Search** (in sidebar):
- α = 0.6 (default, balanced)
- α = 0.3 for technical docs (favor keywords)
- α = 0.8 for creative content (favor semantics)

**Enable Reranking**:
- Check "Use Reranking" ✓
- Requires Cohere API key
- +20% accuracy improvement

### Evaluate Quality

```bash
# Check metrics
cat data/evaluation/results.json

# Target metrics:
# - recall@5: >0.80
# - precision: >0.60
```

If below targets:
1. Check if documents are relevant
2. Try different chunk sizes (500, 1500)
3. Adjust hybrid search alpha
4. Review retrieval failures in logs

### Customize

**Change Models**:

Edit `.env`:
```bash
# Use larger embedding model
TEXT_EMBEDDING_MODEL=text-embedding-3-large

# Use Claude for generation
GENERATION_MODEL=claude-3-opus-20240229
```

**Change Chunking**:

```python
# In your code
from src.document_processing import SectionChunker

chunker = SectionChunker(max_section_size=2000)
```

**Add Caching**:

```python
# Coming soon
```

## Common Issues

### "No documents found"
- Add files to `data/documents/`
- Supported: PDF, DOCX, TXT, MD

### "API key error"
- Check `.env` file exists
- Verify API keys are correct
- Test: `echo $OPENAI_API_KEY`

### "Low retrieval quality"
- Check document relevance
- Increase top_k to 10-20
- Enable reranking
- Try different chunk sizes

### "Slow performance"
- Reduce retrieval_top_k
- Use smaller embedding model
- Enable quantization (in code)

## Production Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to streamlit.io
3. Connect repository
4. Add environment variables (API keys)
5. Deploy!

### Docker

```bash
# Build image
docker build -t multimodal-rag .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e COHERE_API_KEY=$COHERE_API_KEY \
  multimodal-rag
```

## Resources

- [Full Documentation](README.md)
- [API Reference](docs/api.md)
- [Examples](examples/)
- [Troubleshooting](docs/troubleshooting.md)

## Support

- GitHub Issues: <repo-url>/issues
- Discussions: <repo-url>/discussions

---

**Ready to build production RAG? Let's go! 🚀**
