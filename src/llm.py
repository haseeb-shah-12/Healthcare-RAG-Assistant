"""
LLM Provider Module
Handles integration with OpenAI and local LLM models
"""

import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

from src.config import Config

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Manages LLM interactions for answer generation
    """

    def __init__(self, model_name: str = None):
        """
        Initialize LLMProvider

        Args:
            model_name: Name of the model to use (default from config)
        """
        self.model_name = model_name or Config.OPENAI_MODEL
        logger.info(f"Initializing LLMProvider with model: {self.model_name}")

        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
            )
            logger.info(f"Successfully initialized {self.model_name}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise

    def generate_answer(
        self,
        query: str,
        context: str,
        system_prompt: str = None,
    ) -> str:
        """
        Generate answer based on query and retrieved context

        Args:
            query: User question
            context: Retrieved document context
            system_prompt: Custom system prompt (uses default if None)

        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()

        logger.debug(f"Generating answer for query: {query[:100]}...")

        # Build messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]

        try:
            response = self.llm(messages)
            answer = response.content
            logger.debug("Answer generated successfully")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise

    def _get_default_system_prompt(self) -> str:
        """
        Get default system prompt for healthcare RAG

        Returns:
            System prompt string
        """
        return """You are a knowledgeable healthcare assistant trained to provide accurate medical information based on the provided documents.

Guidelines:
1. Always base your answers on the provided context/documents
2. Be clear and accurate in your medical explanations
3. If information is not available in the provided documents, clearly state that
4. Suggest consulting healthcare professionals for medical decisions
5. Maintain a professional and empathetic tone
6. Use clear language, avoiding unnecessary medical jargon where possible
7. Structure answers logically with bullet points or sections when appropriate

Important: This is for informational purposes only and should not replace professional medical advice."""

    def extract_source_references(self, documents: List[Dict]) -> str:
        """
        Format source references from retrieved documents

        Args:
            documents: List of retrieved document chunks

        Returns:
            Formatted source reference string
        """
        sources = set()
        for doc in documents:
            source_info = f"{doc.get('filename', 'Unknown')} - {doc.get('title', 'Untitled')}"
            sources.add(source_info)

        if not sources:
            return "No sources available"

        sources_text = "**Sources:**\n"
        for i, source in enumerate(sources, 1):
            sources_text += f"{i}. {source}\n"

        return sources_text
