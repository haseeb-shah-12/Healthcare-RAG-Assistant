"""
Embedding Generation Module
Converts text to vector embeddings using Sentence Transformers
"""

import logging
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import Config

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text chunks using Sentence Transformers
    """

    def __init__(self, model_name: str = None):
        """
        Initialize EmbeddingGenerator

        Args:
            model_name: Name of Sentence Transformers model (default from config)
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")

        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode single text to embedding

        Args:
            text: Text to encode

        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def encode_texts(
        self, texts: List[str], batch_size: int = None, show_progress: bool = True
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Encode multiple texts to embeddings

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding (default from config)
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (embeddings array, valid indices)
        """
        batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        logger.info(f"Encoding {len(texts)} texts with batch_size={batch_size}")

        valid_texts = []
        valid_indices = []

        # Filter out empty texts
        for idx, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(idx)

        if not valid_texts:
            logger.warning("No valid texts to encode")
            return np.array([]), []

        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )
            embeddings = embeddings.astype(np.float32)
            logger.info(f"Successfully encoded {len(embeddings)} texts")
            return embeddings, valid_indices
        except Exception as e:
            logger.error(f"Error encoding texts: {str(e)}")
            raise

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query to embedding (typically a user question)

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        logger.debug(f"Encoding query: {query[:100]}...")
        return self.encode_text(query)

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "model_type": type(self.model).__name__,
        }
