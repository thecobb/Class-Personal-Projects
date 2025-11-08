"""Multimodal generation using LLMs."""

from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from ..config import settings
from ..document_processing.parsers import ImageElement


class MultimodalGenerator:
    """
    Generator for creating answers from multimodal context.

    Supports both text-only and multimodal (text + images) generation.
    """

    # Default prompt template for RAG
    DEFAULT_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information provided in the context
- If the context contains images, reference them specifically when relevant
- Cite specific passages or image numbers to support your answer
- If the context doesn't contain enough information to answer, say so
- Be concise but comprehensive

Answer:"""

    MULTIMODAL_PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on the provided context, which includes both text and images.

Text Context:
{text_context}

Images are provided below.

Question: {query}

Instructions:
- Answer the question using the text context and images provided
- Analyze the charts, diagrams, or images shown to support your answer
- Reference specific text passages or image numbers for each claim
- If the context doesn't contain enough information to answer, say so
- Be concise but comprehensive

Answer:"""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize multimodal generator.

        Args:
            model: Model name (e.g., 'gpt-4o', 'gpt-4-turbo')
            api_key: OpenAI API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model or settings.generation_model
        self.api_key = api_key or settings.openai_api_key
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Generation will fail.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized MultimodalGenerator with model: {self.model}")

    def generate(
        self,
        query: str,
        context_chunks: List[Any],
        images: Optional[List[ImageElement]] = None,
        prompt_template: Optional[str] = None,
    ) -> str:
        """
        Generate answer from text and optional image context.

        Args:
            query: User question
            context_chunks: Retrieved context chunks
            images: Optional list of image elements
            prompt_template: Custom prompt template

        Returns:
            Generated answer
        """
        has_images = images is not None and len(images) > 0

        # Build text context
        text_context = self._build_text_context(context_chunks)

        # Choose appropriate generation method
        if has_images:
            return self._generate_multimodal(
                query, text_context, images, prompt_template
            )
        else:
            return self._generate_text_only(
                query, text_context, prompt_template
            )

    def _generate_text_only(
        self,
        query: str,
        text_context: str,
        prompt_template: Optional[str] = None,
    ) -> str:
        """Generate answer from text context only."""
        template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE

        # Format prompt
        prompt = template.format(context=text_context, query=query)

        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            logger.debug(f"Generated answer: {answer[:100]}...")

            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _generate_multimodal(
        self,
        query: str,
        text_context: str,
        images: List[ImageElement],
        prompt_template: Optional[str] = None,
    ) -> str:
        """Generate answer from text and image context."""
        template = prompt_template or self.MULTIMODAL_PROMPT_TEMPLATE

        # Format text prompt
        text_prompt = template.format(text_context=text_context, query=query)

        # Build message content with images
        content = [
            {"type": "text", "text": text_prompt}
        ]

        # Add images
        for idx, image in enumerate(images):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image.image_base64}",
                    "detail": "high"  # Use high detail for better analysis
                }
            })

            # Add caption if available
            if image.caption:
                content.append({
                    "type": "text",
                    "text": f"[Image {idx + 1} caption: {image.caption}]"
                })

        try:
            # Call OpenAI Vision API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content
            logger.debug(f"Generated multimodal answer: {answer[:100]}...")

            return answer

        except Exception as e:
            logger.error(f"Error generating multimodal answer: {e}")
            raise

    def _build_text_context(self, context_chunks: List[Any]) -> str:
        """
        Build formatted text context from chunks.

        Args:
            context_chunks: List of context elements (Chunks, strings, etc.)

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, chunk in enumerate(context_chunks):
            # Extract text based on chunk type
            if isinstance(chunk, str):
                text = chunk
            elif hasattr(chunk, "text"):
                text = chunk.text
            else:
                text = str(chunk)

            # Format with index
            context_parts.append(f"[{idx + 1}] {text}")

        return "\n\n".join(context_parts)

    def generate_with_citations(
        self,
        query: str,
        context_chunks: List[Any],
        images: Optional[List[ImageElement]] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer with explicit citation tracking.

        Args:
            query: User question
            context_chunks: Retrieved context chunks
            images: Optional list of images

        Returns:
            Dictionary with 'answer' and 'citations'
        """
        # Enhanced prompt that requests citations
        citation_prompt = self.DEFAULT_PROMPT_TEMPLATE.replace(
            "Answer:",
            """Answer (include [1], [2], etc. to cite sources):"""
        )

        answer = self.generate(query, context_chunks, images, citation_prompt)

        # Parse citations from answer
        import re
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = sorted(set(int(c) for c in citations))

        # Build citation references
        citation_refs = []
        for cite_num in unique_citations:
            idx = cite_num - 1  # Convert to 0-indexed
            if 0 <= idx < len(context_chunks):
                chunk = context_chunks[idx]
                if hasattr(chunk, 'metadata'):
                    citation_refs.append({
                        'number': cite_num,
                        'text': chunk.text[:200] + "...",
                        'metadata': chunk.metadata
                    })

        return {
            'answer': answer,
            'citations': citation_refs,
            'cited_chunks': [context_chunks[i-1] for i in unique_citations if 0 <= i-1 < len(context_chunks)]
        }
