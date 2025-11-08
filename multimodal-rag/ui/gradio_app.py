"""Gradio interface for Multimodal RAG system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
from loguru import logger

from src.config import settings
from src.document_processing import DocumentParser, RecursiveChunker
from src.embeddings import EmbeddingService, ImageDescriber
from src.generation import MultimodalGenerator
from src.retrieval import HybridRetriever, Reranker, VectorStoreFactory


class MultimodalRAGApp:
    """Gradio application for Multimodal RAG."""

    def __init__(self):
        """Initialize the RAG application."""
        logger.info("Initializing Multimodal RAG App")

        # Initialize components
        self.vector_store = VectorStoreFactory.create()
        self.embedding_service = EmbeddingService()
        self.image_describer = ImageDescriber()
        self.retriever = HybridRetriever(self.vector_store, chunks=None)
        self.reranker = Reranker() if settings.use_reranking else None
        self.generator = MultimodalGenerator()

        # Document processing
        self.parser = DocumentParser(extract_images=True, extract_tables=True)
        self.chunker = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )

        logger.info("App initialized successfully")

    def process_document(self, file_path: str, progress=gr.Progress()) -> str:
        """
        Process uploaded document.

        Args:
            file_path: Path to uploaded file
            progress: Gradio progress tracker

        Returns:
            Status message
        """
        try:
            progress(0, desc="Parsing document...")
            logger.info(f"Processing document: {file_path}")

            # Parse document
            doc = self.parser.parse(file_path)

            progress(0.3, desc="Chunking text...")

            # Chunk text
            chunks = self.chunker.chunk(doc)

            progress(0.5, desc="Generating embeddings...")

            # Add to vector store
            self.vector_store.add_chunks(chunks)

            # Update retriever
            self.retriever = HybridRetriever(self.vector_store, chunks)

            progress(1.0, desc="Complete!")

            return f"""✓ Document processed successfully!

- Text elements: {len(doc.text_elements)}
- Images: {len(doc.image_elements)}
- Tables: {len(doc.table_elements)}
- Chunks created: {len(chunks)}
"""

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return f"❌ Error: {str(e)}"

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranking: bool = True,
        progress=gr.Progress()
    ) -> tuple[str, str]:
        """
        Query the RAG system.

        Args:
            question: User question
            top_k: Number of results to retrieve
            use_reranking: Whether to use reranking
            progress: Gradio progress tracker

        Returns:
            Tuple of (answer, context)
        """
        try:
            progress(0, desc="Retrieving context...")
            logger.info(f"Query: {question}")

            # Retrieve chunks
            results = self.retriever.retrieve(question, k=top_k * 2)

            progress(0.4, desc="Reranking...")

            # Rerank if enabled
            if use_reranking and self.reranker:
                results = self.reranker.rerank(question, results, top_n=top_k)
            else:
                results = results[:top_k]

            # Extract chunks
            chunks = [chunk for chunk, score in results]

            progress(0.7, desc="Generating answer...")

            # Generate answer
            answer = self.generator.generate(question, chunks)

            # Format context for display
            context_parts = []
            for i, (chunk, score) in enumerate(results, 1):
                context_parts.append(
                    f"**[{i}] (Score: {score:.3f})**\n{chunk.text}\n"
                )

            context = "\n---\n".join(context_parts)

            progress(1.0, desc="Complete!")

            return answer, context

        except Exception as e:
            logger.error(f"Error querying: {e}")
            return f"❌ Error: {str(e)}", ""

    def get_stats(self) -> str:
        """Get system statistics."""
        try:
            stats = self.vector_store.get_collection_stats()
            return f"""**System Statistics**

- Collection: {stats.get('name', 'N/A')}
- Total chunks: {stats.get('count', 0)}
- Vector DB: {settings.vector_db_type}
- Embedding model: {settings.text_embedding_model}
- Generation model: {settings.generation_model}
"""
        except Exception as e:
            return f"Error getting stats: {e}"


def create_interface() -> gr.Blocks:
    """Create Gradio interface."""
    app = MultimodalRAGApp()

    with gr.Blocks(title="Multimodal RAG System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔍 Multimodal RAG System

        A production-grade RAG system following Jason Liu's methodology with hybrid search,
        reranking, and multimodal capabilities.
        """)

        with gr.Tabs():
            # Document Upload Tab
            with gr.Tab("📄 Document Upload"):
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="Upload Document (PDF, DOCX, TXT, etc.)",
                            file_types=[".pdf", ".docx", ".txt", ".md"]
                        )
                        upload_btn = gr.Button("Process Document", variant="primary")

                    with gr.Column():
                        upload_output = gr.Textbox(
                            label="Status",
                            lines=10,
                            interactive=False
                        )

                upload_btn.click(
                    fn=app.process_document,
                    inputs=[file_input],
                    outputs=[upload_output]
                )

            # Query Tab
            with gr.Tab("💬 Query"):
                with gr.Row():
                    with gr.Column():
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask a question about your documents...",
                            lines=3
                        )

                        with gr.Row():
                            top_k_slider = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Number of results (top-k)"
                            )
                            rerank_checkbox = gr.Checkbox(
                                label="Use reranking",
                                value=True
                            )

                        query_btn = gr.Button("Search", variant="primary")

                    with gr.Column():
                        answer_output = gr.Textbox(
                            label="Answer",
                            lines=10,
                            interactive=False
                        )

                with gr.Row():
                    context_output = gr.Markdown(label="Retrieved Context")

                query_btn.click(
                    fn=app.query,
                    inputs=[question_input, top_k_slider, rerank_checkbox],
                    outputs=[answer_output, context_output]
                )

                # Example questions
                gr.Examples(
                    examples=[
                        ["What are the main topics discussed in the document?"],
                        ["Summarize the key findings"],
                        ["What data is shown in the charts?"],
                    ],
                    inputs=question_input
                )

            # Stats Tab
            with gr.Tab("📊 Statistics"):
                stats_output = gr.Markdown()
                refresh_btn = gr.Button("Refresh Stats")

                refresh_btn.click(
                    fn=app.get_stats,
                    outputs=[stats_output]
                )

                # Load stats on tab open
                interface.load(
                    fn=app.get_stats,
                    outputs=[stats_output]
                )

        gr.Markdown("""
        ---
        **Note:** This system implements Jason Liu's production RAG methodology with:
        - Hybrid search (BM25 + vector)
        - Cohere reranking
        - Multimodal support
        - Systematic evaluation
        """)

    return interface


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/gradio_app.log", rotation="1 day")

    # Create and launch interface
    interface = create_interface()
    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
