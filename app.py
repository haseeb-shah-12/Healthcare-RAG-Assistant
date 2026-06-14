"""
Streamlit Frontend for Healthcare RAG Chatbot
"""

import streamlit as st
import logging
from typing import Dict
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chatbot import HealthcareRAGChatbot
from src.config import Config

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Healthcare RAG Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Styling
st.markdown(
    """
    <style>
        .main {
            padding: 2rem;
        }
        .source-box {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .answer-box {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #0066cc;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_chatbot() -> HealthcareRAGChatbot:
    """Initialize chatbot (cached to avoid reloading)"""
    logger.info("Initializing Healthcare RAG Chatbot")
    chatbot = HealthcareRAGChatbot(verbose=True)
    
    # Load index if available
    if not chatbot.retriever.is_index_loaded():
        st.warning(
            "⚠️ No documents indexed yet. Please run the document ingestion script first:\n"
            "`python ingest.py`"
        )
    
    return chatbot


def display_answer(response: Dict) -> None:
    """Display the generated answer and sources"""
    if response["status"] == "no_results":
        st.info("ℹ️ " + response["answer"])
        return

    # Display answer
    st.markdown("### 📝 Answer")
    st.markdown(f'<div class="answer-box">{response["answer"]}</div>', unsafe_allow_html=True)

    # Display sources
    if response["sources"]:
        st.markdown("### 📚 Sources Used")
        for source in response["sources"]:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{source['filename']}**")
                st.caption(f"📄 {source['title']}")
            with col2:
                st.metric("Similarity", source["similarity"])

    # Display metadata
    with st.expander("📊 Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieved Documents", response["retrieved_docs_count"])
        with col2:
            st.metric("Status", response["status"].upper())


def main():
    """Main Streamlit application"""
    # Header
    st.title("🏥 Healthcare RAG Assistant")
    st.markdown(
        "Ask healthcare questions and get answers based on medical documents "
        "with source attribution."
    )

    # Initialize chatbot
    chatbot = initialize_chatbot()

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        # Check index status
        if chatbot.retriever.is_index_loaded():
            stats = chatbot.retriever.get_index_stats()
            st.success("✅ Database Loaded")
            st.metric("Total Vectors", stats["total_vectors"])
            st.metric("Embedding Dimension", stats["embedding_dimension"])
        else:
            st.error("❌ Database Not Loaded")

        # Settings
        top_k = st.slider("Top Documents to Retrieve", min_value=1, max_value=10, value=5)
        
        st.markdown("---")
        st.markdown("### 📋 About")
        st.info(
            "This chatbot uses RAG (Retrieval-Augmented Generation) to answer "
            "healthcare questions based on medical documents."
        )

        # Disclaimer
        st.markdown("### ⚠️ Disclaimer")
        st.warning(
            "This assistant provides informational content only and should not replace "
            "professional medical advice. Always consult healthcare professionals."
        )

    # Main content
    st.markdown("---")

    # Example questions
    with st.expander("💡 Example Questions"):
        examples = [
            "What are the symptoms of Type 2 Diabetes?",
            "How is hypertension diagnosed?",
            "What are the risk factors for heart disease?",
            "Explain the treatment options for arthritis",
            "What is the causes of chronic kidney disease?",
        ]
        for i, example in enumerate(examples, 1):
            st.caption(f"{i}. {example}")

    # Chat interface
    st.markdown("### 🤖 Ask a Healthcare Question")

    # Use columns for better layout
    col1, col2 = st.columns([4, 1])

    with col1:
        user_question = st.text_input(
            "Your question:",
            placeholder="e.g., What are the symptoms of diabetes?",
            label_visibility="collapsed",
        )

    with col2:
        submit_button = st.button("🔍 Search", use_container_width=True)

    # Process question
    if submit_button and user_question:
        if not chatbot.retriever.is_index_loaded():
            st.error(
                "❌ Database is not loaded. Please ingest documents first using:\n"
                "`python ingest.py`"
            )
        else:
            with st.spinner("🔄 Processing your question..."):
                try:
                    response = chatbot.query(user_question, top_k=top_k)
                    display_answer(response)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    st.error(f"❌ Error processing query: {str(e)}")

    # Chat history (session state)
    st.markdown("---")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history if exists
    if st.session_state.chat_history:
        st.markdown("### 📜 Recent Questions")
        for i, (q, a) in enumerate(st.session_state.chat_history[-5:], 1):
            with st.expander(f"{i}. {q[:50]}..."):
                st.write(a)


if __name__ == "__main__":
    main()
