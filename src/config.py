"""
Configuration Management for Healthcare RAG Assistant
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Configuration class for the Healthcare RAG Assistant"""

    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "data/medical_documents"))
    VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "data/embeddings"))

    # Create directories if they don't exist
    DOCUMENTS_PATH.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 32))

    # Document Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", 100))

    # RAG Configuration
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.5))

    # FAISS Index
    FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "faiss_index.pkl")
    FAISS_INDEX_PATH = VECTOR_DB_PATH / FAISS_INDEX_NAME

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/chatbot.log")

    # Create logs directory
    LOG_DIR = Path(LOG_FILE).parent
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate(cls):
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not set. Please set it in .env file or environment variables."
            )
        logger.info("Configuration validated successfully")

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for debugging"""
        return {
            "DOCUMENTS_PATH": str(cls.DOCUMENTS_PATH),
            "VECTOR_DB_PATH": str(cls.VECTOR_DB_PATH),
            "OPENAI_MODEL": cls.OPENAI_MODEL,
            "EMBEDDING_MODEL": cls.EMBEDDING_MODEL,
            "CHUNK_SIZE": cls.CHUNK_SIZE,
            "CHUNK_OVERLAP": cls.CHUNK_OVERLAP,
            "TOP_K_RETRIEVAL": cls.TOP_K_RETRIEVAL,
        }
