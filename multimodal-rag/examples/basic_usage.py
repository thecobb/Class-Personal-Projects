"""Basic usage examples for Multimodal RAG system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.document_processing import DocumentParser, RecursiveChunker
from src.embeddings import EmbeddingService
from src.generation import MultimodalGenerator
from src.retrieval import HybridRetriever, Reranker, VectorStoreFactory


def example_1_basic_rag():
    """Example 1: Basic RAG pipeline."""
    print("=" * 60)
    print("Example 1: Basic RAG Pipeline")
    print("=" * 60)

    # Initialize
    vector_store = VectorStoreFactory.create()
    embedding_service = EmbeddingService()
    generator = MultimodalGenerator()

    # Parse document
    parser = DocumentParser()
    doc = parser.parse("path/to/your/document.pdf")

    print(f"Parsed document with {len(doc.text_elements)} text elements")

    # Chunk
    chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=200)
    chunks = chunker.chunk(doc)

    print(f"Created {len(chunks)} chunks")

    # Index
    vector_store.add_chunks(chunks)

    # Create retriever
    retriever = HybridRetriever(vector_store, chunks)

    # Query
    question = "What are the main topics?"
    results = retriever.retrieve(question, k=5)

    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(results)} chunks")

    # Generate answer
    chunks_retrieved = [chunk for chunk, score in results]
    answer = generator.generate(question, chunks_retrieved)

    print(f"\nAnswer: {answer}")


def example_2_with_reranking():
    """Example 2: RAG with reranking."""
    print("\n" + "=" * 60)
    print("Example 2: RAG with Reranking")
    print("=" * 60)

    # Setup (same as before)
    vector_store = VectorStoreFactory.create()
    retriever = HybridRetriever(vector_store)
    generator = MultimodalGenerator()

    # Add reranker
    reranker = Reranker()

    question = "What are the main findings?"

    # Retrieve more candidates for reranking
    results = retriever.retrieve(question, k=20)
    print(f"Initial retrieval: {len(results)} chunks")

    # Rerank to top 5
    reranked = reranker.rerank(question, results, top_n=5)
    print(f"After reranking: {len(reranked)} chunks")

    # Generate
    chunks_retrieved = [chunk for chunk, score in reranked]
    answer = generator.generate(question, chunks_retrieved)

    print(f"\nAnswer: {answer}")


def example_3_multimodal():
    """Example 3: Multimodal RAG with images."""
    print("\n" + "=" * 60)
    print("Example 3: Multimodal RAG")
    print("=" * 60)

    from src.embeddings import ImageDescriber
    from src.retrieval import MultiVectorRetriever

    # Setup
    vector_store = VectorStoreFactory.create()
    multi_retriever = MultiVectorRetriever(vector_store)
    image_describer = ImageDescriber()
    generator = MultimodalGenerator()

    # Parse with images
    parser = DocumentParser(extract_images=True)
    doc = parser.parse("path/to/document_with_images.pdf")

    print(f"Found {len(doc.image_elements)} images")

    # Describe images for search
    descriptions = []
    for img in doc.image_elements:
        desc = image_describer.describe_image(img.image_base64)
        descriptions.append(desc)

    # Add images to multi-vector retriever
    multi_retriever.add_images(doc.image_elements, descriptions)

    # Query
    question = "What does the revenue chart show?"
    results = multi_retriever.retrieve(question, k=5, return_originals=True)

    # Separate text and images
    text_chunks = [r for r in results if hasattr(r[0], 'text')]
    image_elements = [r[0] for r in results if hasattr(r[0], 'image_base64')]

    print(f"Retrieved {len(text_chunks)} text chunks and {len(image_elements)} images")

    # Generate with multimodal context
    chunks = [chunk for chunk, score in text_chunks]
    answer = generator.generate(question, chunks, images=image_elements)

    print(f"\nAnswer: {answer}")


def example_4_evaluation():
    """Example 4: Evaluation workflow."""
    print("\n" + "=" * 60)
    print("Example 4: Evaluation")
    print("=" * 60)

    from src.evaluation import SyntheticDataGenerator, RAGEvaluator

    # Setup
    vector_store = VectorStoreFactory.create()
    retriever = HybridRetriever(vector_store)

    # Assume we have chunks from previous examples
    chunks = []  # Your chunks here

    # Generate synthetic data
    generator = SyntheticDataGenerator()
    eval_dataset = generator.generate_dataset(chunks[:10], num_questions_per_chunk=5)

    print(f"Generated {len(eval_dataset)} evaluation questions")

    # Evaluate
    evaluator = RAGEvaluator()

    def retrieve_fn(query, k=5):
        results = retriever.retrieve(query, k=k)
        return [chunk.chunk_id for chunk, score in results]

    metrics = evaluator.evaluate_retrieval(
        test_dataset=eval_dataset,
        retrieve_fn=retrieve_fn,
        k=5
    )

    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")


def example_5_query_routing():
    """Example 5: Intelligent query routing."""
    print("\n" + "=" * 60)
    print("Example 5: Query Routing")
    print("=" * 60)

    from src.generation import QueryRouter

    router = QueryRouter()

    queries = [
        "What are the financial results?",
        "Show me the architecture diagram",
        "What data is in the Q3 revenue table?",
    ]

    for query in queries:
        routing = router.route(query)

        print(f"\nQuery: {query}")
        print(f"  Type: {routing['query_type']}")
        print(f"  Search text: {routing['search_text']}")
        print(f"  Search images: {routing['search_images']}")
        print(f"  Search tables: {routing['search_tables']}")


if __name__ == "__main__":
    print("Multimodal RAG - Usage Examples\n")

    # Run examples
    # Note: Comment out examples that require actual documents

    # example_1_basic_rag()
    # example_2_with_reranking()
    # example_3_multimodal()
    # example_4_evaluation()
    example_5_query_routing()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
