# Website Content Search API

A FastAPI-based semantic search system that can search through website content using AI-powered vector embeddings.

## Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

## Quick Start

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
```

3. Activate virtual environment:
```bash
# Windows
.\venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install fastapi uvicorn python-multipart python-dotenv requests beautifulsoup4 nltk pymilvus sentence-transformers
```

5. Create `.env` file (optional):
```bash
# For Milvus Cloud (optional - app works without it)
MILVUS_URI=your_milvus_uri_here
MILVUS_TOKEN=your_milvus_token_here
```

6. Run the server:
```bash
python main.py
```

Server will start at `http://localhost:8000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

Frontend will start at `http://localhost:3000`

## API Usage

### Search Endpoint
```bash
POST http://localhost:8000/search
Content-Type: application/json

{
  "url": "https://example.com",
  "query": "your search term"
}
```

## Features

- ✅ Web scraping and HTML parsing
- ✅ Text chunking and tokenization
- ✅ Semantic search with AI embeddings
- ✅ Vector database integration (Milvus)
- ✅ Fallback text search (works without AI)
- ✅ CORS enabled for frontend integration

## Notes

- The app works without Milvus credentials (uses text-based search)
- Add Milvus credentials to `.env` for semantic search capabilities
- First run may take time to download AI models
