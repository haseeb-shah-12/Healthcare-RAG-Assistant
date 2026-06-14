# Healthcare RAG Assistant 🏥

An AI-powered healthcare chatbot that answers medical questions by searching through medical documents using **Retrieval-Augmented Generation (RAG)**.

## 🎯 Project Overview

This system retrieves relevant medical information from a document database and uses a Large Language Model (LLM) to generate accurate, evidence-based answers with source citations.

### Architecture Flow

```
Medical PDFs
    ↓
Document Loader
    ↓
Text Chunking
    ↓
Embeddings Model (Sentence Transformers)
    ↓
Vector Database (FAISS)
    ↓
User Question
    ↓
Similarity Search
    ↓
Relevant Chunks Retrieved
    ↓
LLM (OpenAI GPT or Llama 3)
    ↓
Answer + Sources
```

## 🚀 Features

- ✅ **Document Ingestion**: Load and process medical PDFs
- ✅ **Smart Chunking**: Split documents into semantic chunks
- ✅ **Vector Embeddings**: Convert text to embeddings using Sentence Transformers
- ✅ **Semantic Search**: Find relevant medical documents using FAISS
- ✅ **LLM Integration**: OpenAI GPT-4 or open-source alternatives
- ✅ **Source Attribution**: Shows which documents were used
- ✅ **Interactive Chat**: Streamlit-based user interface
- ✅ **Production Ready**: Docker support, error handling, logging

## 📁 Project Structure

```
Healthcare-RAG-Assistant/
│
├── data/
│   ├── medical_documents/
│   │   ├── diabetes.pdf
│   │   ├── heart_disease.pdf
│   │   └── hypertension.pdf
│   └── embeddings/
│       └── faiss_index.pkl
│
├── src/
│   ├── __init__.py
│   ├── ingest.py              # Document loading & chunking
│   ├── embeddings.py          # Embedding generation
│   ├── retriever.py           # Vector search & retrieval
│   ├── llm.py                 # LLM integration
│   ├── chatbot.py             # Main chatbot logic
│   └── config.py              # Configuration
│
├── app.py                     # Streamlit frontend
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
└── README.md                  # This file
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|--------|
| **Language** | Python 3.9+ |
| **LLM** | OpenAI GPT-4 / Llama 3 |
| **Framework** | LangChain |
| **Vector DB** | FAISS |
| **Embeddings** | Sentence Transformers |
| **Frontend** | Streamlit |
| **Document Processing** | PyPDF2 / pdfplumber |
| **Deployment** | Docker / Hugging Face / Render |

## 📋 Prerequisites

- Python 3.9 or higher
- pip or conda
- OpenAI API key (or local Llama model)
- Medical PDF documents

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/haseeb-shah-12/Healthcare-RAG-Assistant.git
cd Healthcare-RAG-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=your_api_key_here
VECTOR_DB_PATH=data/embeddings/
DOCUMENTS_PATH=data/medical_documents/
```

## 🚀 Quick Start

### Step 1: Prepare Medical Documents

Place your medical PDF files in `data/medical_documents/`:

```bash
mkdir -p data/medical_documents
# Add your PDFs here
```

### Step 2: Ingest Documents

```bash
python ingest.py
```

This will:
- Load all PDFs
- Split into chunks
- Generate embeddings
- Save to FAISS index

### Step 3: Run the Chatbot

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

## 💬 Example Queries

- "What are the symptoms of Type 2 Diabetes?"
- "How is hypertension treated?"
- "What are the risk factors for heart disease?"
- "Explain the complications of diabetes"

## 📚 How RAG Works

1. **Query Processing**: User question is converted to embeddings
2. **Semantic Search**: FAISS finds most similar document chunks
3. **Context Building**: Retrieved chunks are formatted as context
4. **LLM Generation**: LLM generates answer based on retrieved context
5. **Source Attribution**: Original documents are cited

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t healthcare-rag .
```

### Run Container

```bash
docker run -p 8501:8501 --env-file .env healthcare-rag
```

### Using Docker Compose

```bash
docker-compose up
```

## 🌐 Deployment Options

### Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Push code to the Space repository
3. Add secrets for API keys
4. Streamlit app runs automatically

### Render

1. Connect GitHub repository
2. Select Python as environment
3. Set start command: `streamlit run app.py`
4. Add environment variables
5. Deploy

## 📊 Performance Metrics

- **Retrieval Time**: < 200ms
- **Answer Generation**: 2-5 seconds
- **Vector Database Size**: ~100MB (for 1000 documents)

## 🔐 Security Best Practices

- ✅ API keys stored in `.env` (never commit)
- ✅ Input validation for user queries
- ✅ Rate limiting for API calls
- ✅ Error handling for sensitive information

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [FAISS Guide](https://github.com/facebookresearch/faiss)
- [Streamlit Docs](https://docs.streamlit.io/)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## ⚠️ Disclaimer

**Important**: This chatbot is for educational purposes and should not replace professional medical advice. Always consult healthcare professionals for medical decisions.

## 👤 Author

[Haseeb Shah] - Healthcare AI Enthusiast

## 🙏 Acknowledgments

- OpenAI for GPT models
- Meta for Llama
- LangChain team
- Facebook Research for FAISS

---

**Questions?** Open an issue on GitHub or contact me directly.
