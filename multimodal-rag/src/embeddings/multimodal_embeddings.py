"""Multimodal embeddings using CLIP and similar models."""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import open_clip
import torch
from loguru import logger
from PIL import Image


class MultimodalEmbeddings(ABC):
    """Abstract base class for multimodal embeddings."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed text into vector space."""
        pass

    @abstractmethod
    def embed_image(self, image: Union[Image.Image, bytes]) -> List[float]:
        """Embed image into vector space."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        pass

    @abstractmethod
    def embed_images(self, images: List[Union[Image.Image, bytes]]) -> List[List[float]]:
        """Embed multiple images."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass


class CLIPEmbeddings(MultimodalEmbeddings):
    """CLIP-based multimodal embeddings for unified text-image retrieval."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize CLIP embeddings.

        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights to use
            device: Device to run model on
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = device

        logger.info(f"Loading CLIP model {model_name} ({pretrained}) on {device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.eval()
        logger.info(f"CLIP model loaded successfully")

    def embed_text(self, text: str) -> List[float]:
        """
        Embed text using CLIP.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            embedding = text_features.cpu().numpy()[0].tolist()
            return embedding

    def embed_image(self, image: Union[Image.Image, bytes]) -> List[float]:
        """
        Embed image using CLIP.

        Args:
            image: PIL Image or image bytes

        Returns:
            Embedding vector
        """
        if isinstance(image, bytes):
            import io

            image = Image.open(io.BytesIO(image))

        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embedding = image_features.cpu().numpy()[0].tolist()
            return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts using CLIP.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            embeddings = text_features.cpu().numpy().tolist()
            return embeddings

    def embed_images(self, images: List[Union[Image.Image, bytes]]) -> List[List[float]]:
        """
        Embed multiple images using CLIP.

        Args:
            images: List of PIL Images or image bytes

        Returns:
            List of embedding vectors
        """
        # Convert bytes to PIL Images if needed
        pil_images = []
        for img in images:
            if isinstance(img, bytes):
                import io

                pil_images.append(Image.open(io.BytesIO(img)))
            else:
                pil_images.append(img)

        with torch.no_grad():
            # Stack preprocessed images
            image_tensors = torch.stack([self.preprocess(img) for img in pil_images]).to(
                self.device
            )
            image_features = self.model.encode_image(image_tensors)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            embeddings = image_features.cpu().numpy().tolist()
            return embeddings

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        # Common CLIP dimensions
        dimension_map = {
            "ViT-B-32": 512,
            "ViT-B-16": 512,
            "ViT-L-14": 768,
            "ViT-L-14-336": 768,
        }
        return dimension_map.get(self.model_name, 512)

    def compute_similarity(self, text_embedding: List[float], image_embedding: List[float]) -> float:
        """
        Compute cosine similarity between text and image embeddings.

        Args:
            text_embedding: Text embedding vector
            image_embedding: Image embedding vector

        Returns:
            Similarity score (0-1)
        """
        text_vec = np.array(text_embedding)
        image_vec = np.array(image_embedding)

        # Cosine similarity
        similarity = np.dot(text_vec, image_vec) / (
            np.linalg.norm(text_vec) * np.linalg.norm(image_vec)
        )
        return float(similarity)
