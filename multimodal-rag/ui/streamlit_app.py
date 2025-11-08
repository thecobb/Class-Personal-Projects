"""Streamlit interface for Multimodal RAG system."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from loguru import logger

from src.config import settings
from src.document_processing import DocumentParser, RecursiveChunker
from src.embeddings import EmbeddingService, ImageDescriber
from src.generation import MultimodalGenerator, QueryRouter
from src.retrieval import HybridRetriever, Reranker, VectorStoreFactory


# Page configuration
st.set_page_config(
    page_title="Multimodal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def initialize_system():
    """Initialize RAG system components (cached)."""
    logger.info("Initializing Multimodal RAG System")

    components = {
        "vector_store": VectorStoreFactory.create(),
        "embedding_service": EmbeddingService(),
        "image_describer": ImageDescriber(),
        "generator": MultimodalGenerator(),
        "query_router": QueryRouter(),
        "reranker": Reranker() if settings.use_reranking else None,
        "parser": DocumentParser(extract_images=True, extract_tables=True),
        "chunker": RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        ),
    }

    logger.info("System initialized successfully")
    return components


def process_document(file, components):
    """Process uploaded document."""
    try:
        # Save uploaded file temporarily
        temp_path = Path(f"./temp/{file.name}")
        temp_path.parent.mkdir(exist_ok=True)

        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Parse document
        with st.spinner("Parsing document..."):
            doc = components["parser"].parse(temp_path)

        # Chunk text
        with st.spinner("Chunking text..."):
            chunks = components["chunker"].chunk(doc)

        # Add to vector store
        with st.spinner("Generating embeddings and indexing..."):
            components["vector_store"].add_chunks(chunks)

        # Update session state
        if "chunks" not in st.session_state:
            st.session_state.chunks = []
        st.session_state.chunks.extend(chunks)

        # Clean up
        temp_path.unlink()

        return True, doc, chunks

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return False, None, None


def query_system(question, components, top_k=5, use_reranking=True, hybrid_alpha=None):
    """Query the RAG system."""
    try:
        # Create retriever with current chunks
        chunks = st.session_state.get("chunks", [])
        if not chunks:
            return None, None, "No documents indexed yet. Please upload documents first."

        retriever = HybridRetriever(components["vector_store"], chunks, alpha=hybrid_alpha)

        # Retrieve
        with st.spinner("Retrieving relevant context..."):
            results = retriever.retrieve(question, k=top_k * 2)

        # Rerank if enabled
        if use_reranking and components["reranker"]:
            with st.spinner("Reranking results..."):
                results = components["reranker"].rerank(question, results, top_n=top_k)
        else:
            results = results[:top_k]

        # Extract chunks
        chunks_retrieved = [chunk for chunk, score in results]

        # Generate answer
        with st.spinner("Generating answer..."):
            answer = components["generator"].generate(question, chunks_retrieved)

        return answer, results, None

    except Exception as e:
        logger.error(f"Error querying: {e}")
        return None, None, str(e)


def main():
    """Main Streamlit application."""
    # Initialize system
    components = initialize_system()

    # Header
    st.title("🔍 Multimodal RAG System")
    st.markdown("""
    Production-grade RAG system following **Jason Liu's methodology** with hybrid search,
    reranking, and multimodal capabilities.
    """)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Retrieval settings
        st.subheader("Retrieval")
        top_k = st.slider("Top-K Results", min_value=1, max_value=20, value=5)
        use_reranking = st.checkbox("Use Reranking", value=True)
        hybrid_alpha = st.slider(
            "Hybrid Search α",
            min_value=0.0,
            max_value=1.0,
            value=settings.hybrid_search_alpha,
            help="Weight for semantic search (1.0 = pure vector, 0.0 = pure BM25)"
        )

        # Generation settings
        st.subheader("Generation")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.temperature,
            step=0.1
        )

        st.divider()

        # System info
        st.subheader("📊 System Info")
        stats = components["vector_store"].get_collection_stats()
        st.metric("Indexed Chunks", stats.get("count", 0))
        st.caption(f"Vector DB: {settings.vector_db_type}")
        st.caption(f"Embedding: {settings.text_embedding_model}")
        st.caption(f"LLM: {settings.generation_model}")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📄 Documents", "💬 Query", "📈 Evaluation"])

    # Tab 1: Document Upload
    with tab1:
        st.header("Document Management")

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=["pdf", "docx", "txt", "md"],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, TXT, or Markdown files"
            )

            if uploaded_files:
                process_btn = st.button("Process Documents", type="primary")

                if process_btn:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, file in enumerate(uploaded_files):
                        status_text.text(f"Processing {file.name}...")

                        success, doc, chunks = process_document(file, components)

                        if success:
                            st.success(f"✓ {file.name} processed successfully!")
                            st.write(f"- Text elements: {len(doc.text_elements)}")
                            st.write(f"- Images: {len(doc.image_elements)}")
                            st.write(f"- Tables: {len(doc.table_elements)}")
                            st.write(f"- Chunks created: {len(chunks)}")
                        else:
                            st.error(f"✗ Failed to process {file.name}")

                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    status_text.text("All documents processed!")

        with col2:
            st.subheader("Processing Stats")
            if "chunks" in st.session_state:
                total_chunks = len(st.session_state.chunks)
                st.metric("Total Chunks", total_chunks)
            else:
                st.info("No documents processed yet")

    # Tab 2: Query Interface
    with tab2:
        st.header("Ask Questions")

        # Query input
        question = st.text_area(
            "Your Question",
            placeholder="Ask a question about your documents...",
            height=100,
            key="question_input"
        )

        # Example questions
        with st.expander("📝 Example Questions"):
            examples = [
                "What are the main topics discussed in the document?",
                "Summarize the key findings",
                "What data is shown in the charts?",
                "What are the conclusions?",
            ]
            for ex in examples:
                if st.button(ex, key=f"ex_{ex[:20]}"):
                    st.session_state.question_input = ex

        # Search button
        if st.button("🔍 Search", type="primary", disabled=not question):
            answer, results, error = query_system(
                question,
                components,
                top_k=top_k,
                use_reranking=use_reranking,
                hybrid_alpha=hybrid_alpha
            )

            if error:
                st.error(f"Error: {error}")
            elif answer:
                # Display answer
                st.subheader("💡 Answer")
                st.write(answer)

                # Display retrieved context
                st.subheader("📚 Retrieved Context")

                for idx, (chunk, score) in enumerate(results, 1):
                    with st.expander(
                        f"**[{idx}]** Score: {score:.3f} | "
                        f"{chunk.metadata.get('source', 'Unknown source')}"
                    ):
                        st.write(chunk.text)
                        st.caption(f"Chunk ID: {chunk.chunk_id}")

                # Save to history
                if "query_history" not in st.session_state:
                    st.session_state.query_history = []

                st.session_state.query_history.append({
                    "question": question,
                    "answer": answer,
                    "num_sources": len(results),
                })

        # Query history
        if "query_history" in st.session_state and st.session_state.query_history:
            st.divider()
            st.subheader("📜 Query History")

            for idx, item in enumerate(reversed(st.session_state.query_history[-5:])):
                with st.expander(f"Q: {item['question'][:50]}..."):
                    st.write(f"**Answer:** {item['answer']}")
                    st.caption(f"Sources used: {item['num_sources']}")

    # Tab 3: Evaluation
    with tab3:
        st.header("System Evaluation")

        st.info("""
        Following Jason Liu's methodology:
        1. Start with synthetic data (5 questions/chunk)
        2. Target 97% recall on synthetic data
        3. Measure retrieval before generation
        4. Segment by query type
        5. Iterate based on feedback
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Metrics")

            if "evaluation_metrics" in st.session_state:
                metrics = st.session_state.evaluation_metrics

                st.metric("Retrieval Precision", f"{metrics.get('precision', 0):.2%}")
                st.metric("Retrieval Recall", f"{metrics.get('recall', 0):.2%}")
                st.metric("Faithfulness", f"{metrics.get('faithfulness', 0):.2%}")
            else:
                st.info("Run evaluation to see metrics")

        with col2:
            st.subheader("Actions")

            if st.button("Generate Synthetic Dataset"):
                st.warning("Synthetic dataset generation would run here")

            if st.button("Run Evaluation"):
                st.warning("Evaluation would run here")

        # Feedback collection
        st.divider()
        st.subheader("💬 Provide Feedback")

        feedback_col1, feedback_col2 = st.columns([3, 1])

        with feedback_col1:
            feedback_text = st.text_input(
                "How can we improve?",
                placeholder="Your feedback helps us improve the system..."
            )

        with feedback_col2:
            if st.button("Submit Feedback"):
                if feedback_text:
                    st.success("Thank you for your feedback!")
                    # In production, save feedback to database
                else:
                    st.warning("Please enter feedback first")

    # Footer
    st.divider()
    st.caption("""
    **Multimodal RAG System** | Built with Jason Liu's methodology |
    Features: Hybrid Search • Reranking • Multimodal Support • Systematic Evaluation
    """)


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/streamlit_app.log", rotation="1 day")

    # Run app
    main()
