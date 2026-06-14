"""
Main Chatbot Module
Orchestrates RAG pipeline: retrieval + generation
"""

import logging
from typing import Dict, List
from src.ingest import DocumentIngester
from src.embeddings import EmbeddingGenerator
from src.retriever import VectorRetriever
from src.llm import LLMProvider
from src.config import Config

logger = logging.getLogger(__name__)


class HealthcareRAGChatbot:
    """
    Main chatbot class that orchestrates the complete RAG pipeline
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize the Healthcare RAG Chatbot

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        logger.info("Initializing HealthcareRAGChatbot")

        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.retriever = VectorRetriever(self.embedding_generator)
        self.llm_provider = LLMProvider()
        self.document_ingester = DocumentIngester()

        # Try to load existing index
        if not self.retriever.load_index():
            logger.warning(
                "No existing FAISS index found. "
                "Please run ingest_documents() first."
            )

        logger.info("HealthcareRAGChatbot initialized successfully")

    def ingest_documents(self) -> bool:
        """
        Ingest and index all medical documents

        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting document ingestion")

        try:
            # Load and chunk documents
            documents = self.document_ingester.ingest_and_chunk()

            if not documents:
                logger.error("No documents were ingested")
                return False

            # Build FAISS index
            self.retriever.build_index(documents)
            logger.info("Document ingestion completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during document ingestion: {str(e)}")
            return False

    def query(self, user_question: str, top_k: int = None) -> Dict:
        """
        Answer a user question using RAG pipeline

        Args:
            user_question: The user's medical question
            top_k: Number of top documents to retrieve

        Returns:
            Dictionary containing answer, sources, and metadata
        """
        top_k = top_k or Config.TOP_K_RETRIEVAL

        logger.info(f"Processing query: {user_question[:100]}...")

        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.search(user_question, k=top_k)

        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "answer": (
                    "I couldn't find relevant information in my database to answer "
                    "your question. Please consult a healthcare professional."
                ),
                "sources": [],
                "query": user_question,
                "status": "no_results",
            }

        # Step 2: Build context from retrieved documents
        context = self._build_context(retrieved_docs)

        if self.verbose:
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents")

        # Step 3: Generate answer using LLM
        try:
            answer = self.llm_provider.generate_answer(user_question, context)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            answer = (
                "I encountered an error while generating the answer. "
                "Please try again."
            )

        # Step 4: Format response with sources
        response = {
            "answer": answer,
            "sources": self._format_sources(retrieved_docs),
            "query": user_question,
            "retrieved_docs_count": len(retrieved_docs),
            "status": "success",
        }

        if self.verbose:
            response["retrieved_context"] = context
            response["retrieved_documents"] = retrieved_docs

        logger.info("Query processing completed")
        return response

    def _build_context(self, retrieved_docs: List[Dict]) -> str:
        """
        Build context string from retrieved documents

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            context_part = f"""
Document {i}:
Title: {doc.get('title', 'Unknown')}
Source: {doc.get('filename', 'Unknown')}
Similarity: {doc.get('similarity', 0):.2%}

Content:
{doc.get('content', '')}
"""
            context_parts.append(context_part)

        return "\n".join(context_parts)

    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Format sources from retrieved documents

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            List of formatted source dictionaries
        """
        sources = []
        seen_files = set()

        for doc in retrieved_docs:
            file_key = doc.get("filename", "Unknown")

            if file_key not in seen_files:
                sources.append(
                    {
                        "title": doc.get("title", "Unknown"),
                        "filename": file_key,
                        "similarity": f"{doc.get('similarity', 0):.2%}",
                    }
                )
                seen_files.add(file_key)

        return sources

    def get_stats(self) -> Dict:
        """
        Get statistics about the chatbot and index

        Returns:
            Dictionary with statistics
        """
        stats = {
            "index_loaded": self.retriever.is_index_loaded(),
            "config": Config.to_dict(),
        }

        if self.retriever.is_index_loaded():
            stats["index_stats"] = self.retriever.get_index_stats()

        return stats
