"""
Vector Retriever Module
Handles FAISS-based semantic search and document retrieval
"""

import logging
from typing import List, Dict
import numpy as np
import faiss
import pickle
from pathlib import Path

from src.config import Config
from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class VectorRetriever:
    """
    Manages FAISS vector database and semantic search
    """

    def __init__(self, embedding_generator: EmbeddingGenerator = None):
        """
        Initialize VectorRetriever

        Args:
            embedding_generator: EmbeddingGenerator instance (creates new if None)
        """
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.faiss_index = None
        self.index_path = Config.FAISS_INDEX_PATH
        self.metadata_path = Config.VECTOR_DB_PATH / "metadata.pkl"
        self.chunk_metadata = []
        self.embedding_dim = self.embedding_generator.embedding_dim

        logger.info("VectorRetriever initialized")

    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """
        Build FAISS index from document chunks

        Args:
            documents: List of document chunks with metadata
        """
        logger.info(f"Building FAISS index for {len(documents)} documents")

        if not documents:
            logger.warning("No documents provided for indexing")
            return

        # Extract content from documents
        contents = [doc["content"] for doc in documents]

        # Encode all documents
        embeddings, valid_indices = self.embedding_generator.encode_texts(contents)

        if len(embeddings) == 0:
            logger.error("Failed to generate embeddings")
            return

        # Create FAISS index
        logger.info(f"Creating FAISS index with {len(embeddings)} embeddings")
        self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        self.faiss_index.add(embeddings)

        # Store metadata for valid documents only
        self.chunk_metadata = [documents[i] for i in valid_indices]

        # Save index and metadata
        self.save_index()
        logger.info(f"Successfully built and saved FAISS index with {len(embeddings)} vectors")

    def save_index(self) -> None:
        """Save FAISS index and metadata to disk"""
        try:
            faiss.write_index(self.faiss_index, str(self.index_path))
            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.chunk_metadata, f)
            logger.info(f"Index saved to {self.index_path}")
            logger.info(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            raise

    def load_index(self) -> bool:
        """
        Load FAISS index and metadata from disk

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.index_path.exists() or not self.metadata_path.exists():
                logger.warning(f"Index files not found at {self.index_path}")
                return False

            self.faiss_index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as f:
                self.chunk_metadata = pickle.load(f)

            logger.info(f"Loaded FAISS index with {len(self.chunk_metadata)} vectors")
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False

    def search(
        self, query: str, k: int = None, similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Search for similar documents

        Args:
            query: Query text
            k: Number of top results to return (default from config)
            similarity_threshold: Minimum similarity score (default from config)

        Returns:
            List of retrieved documents with similarity scores
        """
        k = k or Config.TOP_K_RETRIEVAL
        similarity_threshold = similarity_threshold or Config.SIMILARITY_THRESHOLD

        if self.faiss_index is None:
            logger.error("FAISS index not loaded")
            return []

        logger.debug(f"Searching for: {query[:100]}...")

        # Encode query
        query_embedding = self.embedding_generator.encode_query(query)
        query_embedding = np.array([query_embedding]).astype(np.float32)

        # Search FAISS
        distances, indices = self.faiss_index.search(query_embedding, k)
        distances = distances[0]  # Get first (and only) result

        # Convert L2 distances to similarity scores
        # L2 distance to cosine similarity approximation
        similarities = 1 / (1 + distances)

        results = []
        for idx, similarity in zip(indices, similarities):
            if similarity >= similarity_threshold and idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx]
                results.append(
                    {
                        "content": metadata["content"],
                        "source": metadata.get("source", "Unknown"),
                        "filename": metadata.get("filename", "Unknown"),
                        "title": metadata.get("title", "Unknown"),
                        "chunk_id": metadata.get("chunk_id", 0),
                        "similarity": float(similarity),
                    }
                )

        logger.info(f"Found {len(results)} relevant documents")
        return results

    def get_index_stats(self) -> Dict:
        """
        Get statistics about the index

        Returns:
            Dictionary with index statistics
        """
        if self.faiss_index is None:
            return {"status": "No index loaded"}

        return {
            "total_vectors": self.faiss_index.ntotal,
            "embedding_dimension": self.embedding_dim,
            "vector_data_size_mb": (self.faiss_index.ntotal * self.embedding_dim * 4)
            / (1024 * 1024),
            "index_type": type(self.faiss_index).__name__,
        }

    def is_index_loaded(self) -> bool:
        """Check if FAISS index is loaded"""
        return self.faiss_index is not None and self.faiss_index.ntotal > 0
