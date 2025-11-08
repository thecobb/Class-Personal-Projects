"""Example script demonstrating the complete multimodal RAG pipeline."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.config import settings
from src.document_processing import DocumentParser, RecursiveChunker
from src.embeddings import EmbeddingService, ImageDescriber
from src.evaluation import RAGEvaluator, SyntheticDataGenerator
from src.generation import MultimodalGenerator, QueryRouter
from src.retrieval import HybridRetriever, Reranker, VectorStoreFactory


def main():
    """Run the complete multimodal RAG pipeline."""
    logger.info("Starting Multimodal RAG Pipeline")

    # Step 1: Initialize components
    logger.info("=" * 60)
    logger.info("Step 1: Initializing Components")
    logger.info("=" * 60)

    vector_store = VectorStoreFactory.create()
    embedding_service = EmbeddingService()
    image_describer = ImageDescriber()
    generator = MultimodalGenerator()
    query_router = QueryRouter()
    reranker = Reranker() if settings.use_reranking else None

    # Document processing
    parser = DocumentParser(extract_images=True, extract_tables=True)
    chunker = RecursiveChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    logger.info("✓ Components initialized")

    # Step 2: Process documents
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Processing Documents")
    logger.info("=" * 60)

    # Example: Process a document from data directory
    data_dir = Path("./data/documents")
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("Creating sample directory structure...")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Please add documents to ./data/documents/ and re-run")
        return

    # Process all documents in directory
    documents = list(data_dir.glob("*.pdf")) + list(data_dir.glob("*.docx"))

    if not documents:
        logger.warning("No documents found in ./data/documents/")
        logger.info("Please add PDF or DOCX files and re-run")
        return

    all_chunks = []

    for doc_path in documents:
        logger.info(f"\nProcessing: {doc_path.name}")

        # Parse document
        doc = parser.parse(doc_path)
        logger.info(
            f"  - Text elements: {len(doc.text_elements)}, "
            f"Images: {len(doc.image_elements)}, "
            f"Tables: {len(doc.table_elements)}"
        )

        # Chunk text
        chunks = chunker.chunk(doc)
        logger.info(f"  - Created {len(chunks)} chunks")

        all_chunks.extend(chunks)

    logger.info(f"\n✓ Total chunks: {len(all_chunks)}")

    # Step 3: Index chunks
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Indexing Chunks")
    logger.info("=" * 60)

    vector_store.add_chunks(all_chunks)
    logger.info("✓ Chunks indexed in vector store")

    # Step 4: Create retriever
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Setting Up Retrieval")
    logger.info("=" * 60)

    retriever = HybridRetriever(
        vector_store=vector_store,
        chunks=all_chunks,
        alpha=settings.hybrid_search_alpha,
    )
    logger.info(f"✓ Hybrid retriever created (α={settings.hybrid_search_alpha})")

    # Step 5: Generate synthetic evaluation data
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Generating Synthetic Evaluation Data")
    logger.info("=" * 60)

    synthetic_gen = SyntheticDataGenerator()

    # Generate questions for first 10 chunks
    eval_dataset = synthetic_gen.generate_dataset(all_chunks[:10], num_questions_per_chunk=3)
    logger.info(f"✓ Generated {len(eval_dataset)} synthetic question-answer pairs")

    # Save dataset
    import json

    eval_path = Path("./data/evaluation/synthetic_dataset.json")
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_path, "w") as f:
        json.dump(eval_dataset, f, indent=2)

    logger.info(f"✓ Saved evaluation dataset to {eval_path}")

    # Step 6: Example queries
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Running Example Queries")
    logger.info("=" * 60)

    example_questions = [
        "What are the main topics discussed in the document?",
        "Summarize the key findings",
        "What conclusions are presented?",
    ]

    for question in example_questions:
        logger.info(f"\nQuestion: {question}")

        # Route query
        routing = query_router.route(question)
        logger.info(f"  Query type: {routing['query_type']}")

        # Retrieve
        results = retriever.retrieve(question, k=10)

        # Rerank if available
        if reranker:
            results = reranker.rerank(question, results, top_n=5)

        logger.info(f"  Retrieved {len(results)} chunks")

        # Generate answer
        chunks = [chunk for chunk, score in results]
        answer = generator.generate(question, chunks)

        logger.info(f"  Answer: {answer[:200]}...")

    # Step 7: Run evaluation
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Running Evaluation")
    logger.info("=" * 60)

    evaluator = RAGEvaluator(
        synthetic_generator=synthetic_gen,
    )

    # Define retrieval function for evaluation
    def retrieve_fn(query, k=5):
        results = retriever.retrieve(query, k=k)
        if reranker:
            results = reranker.rerank(query, results, top_n=k)
        return [chunk.chunk_id for chunk, score in results]

    # Evaluate retrieval
    retrieval_metrics = evaluator.evaluate_retrieval(
        test_dataset=eval_dataset[:20],  # Use subset for speed
        retrieve_fn=retrieve_fn,
        k=5,
    )

    logger.info("\n📊 Retrieval Metrics:")
    for metric, value in retrieval_metrics.items():
        logger.info(f"  {metric}: {value:.3f}")

    # Save results
    results_path = Path("./data/evaluation/results.json")
    evaluator.save_results(retrieval_metrics, results_path)
    logger.info(f"\n✓ Saved results to {results_path}")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)

    stats = vector_store.get_collection_stats()
    logger.info(f"Vector Store: {stats}")
    logger.info(f"Total Documents Processed: {len(documents)}")
    logger.info(f"Total Chunks Indexed: {stats.get('count', 0)}")
    logger.info(f"Recall@5: {retrieval_metrics.get('recall@5', 0):.1%}")

    logger.info("\nNext steps:")
    logger.info("1. Review evaluation results in ./data/evaluation/")
    logger.info("2. Launch UI: streamlit run ui/streamlit_app.py")
    logger.info("3. Iterate based on metrics")


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/pipeline.log", rotation="1 day")

    # Run pipeline
    try:
        main()
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)
