"""
Document Ingestion and Chunking Module
Handles loading, processing, and chunking of medical PDFs
"""

import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import PyPDF2
import pdfplumber

from src.config import Config

logger = logging.getLogger(__name__)


class DocumentIngester:
    """
    Loads and chunks medical documents for RAG system
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize DocumentIngester

        Args:
            chunk_size: Size of text chunks (default from config)
            chunk_overlap: Overlap between chunks (default from config)
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.documents_path = Config.DOCUMENTS_PATH
        logger.info(
            f"Initialized DocumentIngester with chunk_size={self.chunk_size}, "
            f"overlap={self.chunk_overlap}"
        )

    def load_all_documents(self) -> List[Dict[str, str]]:
        """
        Load all PDF files from documents directory

        Returns:
            List of documents with metadata
        """
        documents = []
        pdf_files = list(self.documents_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_path}")
            return documents

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
            try:
                text = self._extract_text_from_pdf(pdf_file)
                if text.strip():
                    documents.append(
                        {
                            "title": pdf_file.stem,
                            "source": str(pdf_file),
                            "content": text,
                            "filename": pdf_file.name,
                        }
                    )
                    logger.info(f"Successfully loaded {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {str(e)}")
                continue

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            Extracted text content
        """
        text = ""

        try:
            # Try using pdfplumber first (better for structured PDFs)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logger.debug(f"pdfplumber failed for {pdf_path}, trying PyPDF2: {e}")

        # Fallback to PyPDF2
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
            raise

    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split documents into overlapping chunks

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunked documents with metadata
        """
        chunked_docs = []

        for doc in tqdm(documents, desc="Chunking documents"):
            chunks = self._create_chunks(doc["content"])

            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    {
                        "title": doc["title"],
                        "source": doc["source"],
                        "filename": doc["filename"],
                        "content": chunk,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                    }
                )

        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        return chunked_docs

    def _create_chunks(self, text: str) -> List[str]:
        """
        Create overlapping chunks from text

        Args:
            text: Text to chunk

        Returns:
            List of text chunks
        """
        chunks = []
        text_length = len(text)
        start_idx = 0

        while start_idx < text_length:
            end_idx = start_idx + self.chunk_size
            chunk = text[start_idx:end_idx]

            # Try to break at sentence boundary
            if end_idx < text_length:
                # Find last period within reasonable range
                last_period = chunk.rfind(".")
                if last_period > self.chunk_size * 0.7:  # Ensure chunk isn't too small
                    end_idx = start_idx + last_period + 1
                else:
                    # Try to break at newline
                    last_newline = chunk.rfind("\n")
                    if last_newline > self.chunk_size * 0.7:
                        end_idx = start_idx + last_newline + 1

                chunk = text[start_idx:end_idx]

            chunks.append(chunk.strip())
            start_idx = end_idx - self.chunk_overlap

        return [chunk for chunk in chunks if chunk]  # Remove empty chunks

    def ingest_and_chunk(self) -> List[Dict[str, str]]:
        """
        Complete pipeline: load and chunk all documents

        Returns:
            List of chunked documents
        """
        logger.info("Starting document ingestion pipeline")
        documents = self.load_all_documents()

        if not documents:
            logger.warning("No documents loaded, returning empty list")
            return []

        chunked_documents = self.chunk_documents(documents)
        logger.info("Document ingestion pipeline completed")
        return chunked_documents
