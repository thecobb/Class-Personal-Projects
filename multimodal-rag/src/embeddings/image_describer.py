"""Image description generation using vision models."""

import base64
from typing import List, Optional

from loguru import logger
from openai import OpenAI

from ..config import settings


class ImageDescriber:
    """Generate detailed descriptions of images for RAG retrieval."""

    # Optimized prompt for RAG-oriented image descriptions
    DEFAULT_PROMPT = """Analyze this image in detail for search and retrieval purposes. Provide:

1. **General Overview**: What is the primary subject and scene?
2. **Objects and Entities**: List all visible objects, people, and entities
3. **Actions and Interactions**: What is happening in the image?
4. **Environment and Background**: Describe the setting and context
5. **Text Content**: Extract and list all visible text, labels, and numbers
6. **Spatial Relationships**: How are elements positioned relative to each other?
7. **Distinctive Features**: What unique characteristics would help identify this image?

For charts and diagrams specifically:
- Chart type (bar, line, pie, scatter, etc.)
- Axis labels and scales
- Data trends and key insights
- Title and legend information

Be comprehensive but concise. Focus on searchable content."""

    CHART_PROMPT = """Analyze this chart/graph/diagram in detail:

1. **Chart Type**: Identify the type of visualization
2. **Title and Labels**: Extract the title, axis labels, and legend
3. **Data Points**: Describe the key data points and values shown
4. **Trends and Patterns**: What trends or patterns are visible?
5. **Comparisons**: What comparisons does the chart make?
6. **Key Insights**: What are the main takeaways from this visualization?
7. **Additional Context**: Any additional text, annotations, or context

Provide a detailed description that captures all information someone would need to understand and search for this chart."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: int = 500,
    ):
        """
        Initialize image describer.

        Args:
            model: Vision model name (e.g., 'gpt-4o', 'gpt-4-vision-preview')
            api_key: OpenAI API key
            max_tokens: Maximum tokens in generated description
        """
        self.model = model or settings.vision_model
        self.api_key = api_key or settings.openai_api_key
        self.max_tokens = max_tokens

        if not self.api_key:
            logger.warning("No OpenAI API key provided. Image description will fail.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized ImageDescriber with model: {self.model}")

    def describe_image(
        self,
        image_base64: str,
        prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Generate a detailed description of an image.

        Args:
            image_base64: Base64-encoded image
            prompt: Custom prompt (uses default if not provided)
            context: Additional context from surrounding document text

        Returns:
            Detailed image description
        """
        try:
            # Build the prompt
            full_prompt = prompt or self.DEFAULT_PROMPT

            if context:
                full_prompt = f"{full_prompt}\n\nDocument Context:\n{context}"

            # Call vision API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            description = response.choices[0].message.content
            logger.debug(f"Generated image description: {description[:100]}...")
            return description

        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            raise

    def describe_chart(self, image_base64: str, context: Optional[str] = None) -> str:
        """
        Generate a description specifically optimized for charts and diagrams.

        Args:
            image_base64: Base64-encoded chart/diagram image
            context: Additional context from surrounding document text

        Returns:
            Detailed chart description
        """
        return self.describe_image(image_base64, prompt=self.CHART_PROMPT, context=context)

    def describe_images_batch(
        self,
        images_base64: List[str],
        prompts: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate descriptions for multiple images.

        Args:
            images_base64: List of base64-encoded images
            prompts: Optional list of custom prompts
            contexts: Optional list of contexts

        Returns:
            List of image descriptions
        """
        descriptions = []

        for idx, image_b64 in enumerate(images_base64):
            prompt = prompts[idx] if prompts and idx < len(prompts) else None
            context = contexts[idx] if contexts and idx < len(contexts) else None

            try:
                description = self.describe_image(image_b64, prompt=prompt, context=context)
                descriptions.append(description)
            except Exception as e:
                logger.error(f"Error describing image {idx}: {e}")
                descriptions.append(f"[Error describing image: {str(e)}]")

        return descriptions

    def classify_image_type(self, image_base64: str) -> str:
        """
        Classify image type for routing to specialized prompts.

        Args:
            image_base64: Base64-encoded image

        Returns:
            Image type: 'photo', 'chart', 'diagram', or 'document'
        """
        classification_prompt = """Classify this image into one of these categories:
- photo: Photographs of people, products, scenes
- chart: Charts, graphs, data visualizations
- diagram: Flowcharts, architecture diagrams, schematics
- document: Forms, invoices, receipts, text documents

Return only the category name, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": classification_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=10,
            )

            image_type = response.choices[0].message.content.strip().lower()
            logger.debug(f"Classified image as: {image_type}")
            return image_type

        except Exception as e:
            logger.error(f"Error classifying image: {e}")
            return "photo"  # Default fallback
