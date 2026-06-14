"""
Document Ingestion Script
Run this to process and index all medical documents
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.chatbot import HealthcareRAGChatbot
from src.config import Config

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion workflow"""
    logger.info("=" * 60)
    logger.info("Healthcare RAG Assistant - Document Ingestion")
    logger.info("=" * 60)

    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        sys.exit(1)

    logger.info(f"Configuration: {Config.to_dict()}")

    # Check if documents exist
    pdf_files = list(Config.DOCUMENTS_PATH.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {Config.DOCUMENTS_PATH}")
        logger.info("Please add medical PDF files to the data/medical_documents/ directory")
        sys.exit(1)

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Initialize and run ingestion
    try:
        chatbot = HealthcareRAGChatbot()
        success = chatbot.ingest_documents()

        if success:
            logger.info("=" * 60)
            logger.info("✅ Document ingestion completed successfully!")
            logger.info("=" * 60)

            # Display statistics
            stats = chatbot.get_stats()
            if "index_stats" in stats:
                logger.info(f"Index Statistics: {stats['index_stats']}")

            logger.info("\nNext steps:")
            logger.info("1. Run the Streamlit app: streamlit run app.py")
            logger.info("2. Ask healthcare questions in the web interface")
        else:
            logger.error("Document ingestion failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
