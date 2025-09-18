from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
from bs4 import BeautifulSoup
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
print('dwawd')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f".env file exists: {os.path.exists('.env')}")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)


def initialize_ai_components():
    try:
        logger.info("Starting AI components initialization...")
        
        milvus_uri = os.getenv("MILVUS_URI")
        milvus_token = os.getenv("MILVUS_TOKEN")
        
        if not milvus_uri or not milvus_token:
            return None, None
                
        try:
            connections.connect(
                alias="default",
                uri=milvus_uri,
                token=milvus_token,
                timeout=10  
            )
            logger.info("Connected to Milvus Cloud")
        except Exception as conn_error:
            logger.error(f"Failed to connect to Milvus: {str(conn_error)}")
            logger.info("Proceeding without Milvus - semantic search will be disabled")
            return None, None
        
        # Collection name
        collection_name = "test"
        
        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
        ]
        
        schema = CollectionSchema(fields, f"Collection for {collection_name}")
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            logger.info(f"Using existing Milvus collection: {collection_name}")
        else:
            collection = Collection(collection_name, schema)
            logger.info(f"Created new Milvus collection: {collection_name}")
        
        # Create index for vector field
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("AI components initialization completed successfully!")
        return collection, model
        
    except Exception as e:
        logger.error(f"Error initializing AI components: {str(e)}")
        logger.info("Continuing without AI components - basic functionality will be available")
        return None, None

try:
    milvus_collection, sentence_model = initialize_ai_components()
    if milvus_collection is None or sentence_model is None:
        logger.warning("AI components not available -")
    else:
        logger.info("AI components initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AI components: {str(e)}")
    milvus_collection = None
    sentence_model = None

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchRequest(BaseModel):
    url: str
    query: str

class SearchResult(BaseModel):
    content: str
    relevance_score: float
    chunk_index: int

class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_chunks: int
    query: str
    url: str

async def fetch_html_content(url: str) -> str:

    try:
        response = requests.get(url, timeout=30)
        return response.text

    except Exception as e:
        logger.error(f"Unexpected error fetching URL {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def parse_html_content(html_content: str) -> str:

    try:
        soup = BeautifulSoup(html_content, 'lxml')
        
        for script in soup(["script", "style"]):
            script.decompose()
        # Get text content
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except Exception as e:
        logger.error(f"Error parsing HTML: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing HTML content: {str(e)}")

def tokenize_and_chunk_content(text: str) -> list[str]:

    try:
        max_tokens = 500
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        chunks = []
        current_chunk = ""
        current_token_count = 0
        
        for sentence in sentences:
            # Count tokens in the current sentence
            sentence_tokens = word_tokenize(sentence)
            sentence_token_count = len(sentence_tokens)
            
            # If adding this sentence would exceed the limit, save current chunk
            if current_token_count + sentence_token_count > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_token_count = sentence_token_count
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_token_count += sentence_token_count
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error tokenizing content: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error tokenizing content: {str(e)}")

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    try:
        if sentence_model is None:
            raise HTTPException(status_code=503, detail="AI components not available")
        embeddings = sentence_model.encode(texts)
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

def index_chunks_to_milvus(chunks: List[str], url: str) -> None:
    try:
        if milvus_collection is None:
            logger.warning("Milvus not available - skipping indexing")
            return
                    
        embeddings = generate_embeddings(chunks)
        
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            data.append({
                "url": url,
                "chunk_index": i,
                "content": chunk[:5000],  
                "embedding": embedding
            })
        
        milvus_collection.insert(data)
        milvus_collection.flush()
        logger.info(f"Successfully indexed {len(data)} vectors to Milvus")
        
    except Exception as e:
        logger.error(f"Error indexing to Milvus: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error indexing to Milvus: {str(e)}")

def search_similar_chunks(query: str, url: str, top_k: int = 10) -> List[Dict[str, Any]]:
    try:
        if milvus_collection is None:
            return []
        
        # Generate embedding for the query
        query_embedding = generate_embeddings([query])[0]
        
        # Load collection and search
        milvus_collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Search in Milvus
        search_results = milvus_collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=f'url == "{url}"',  
            output_fields=["url", "chunk_index", "content"]
        )
        
        # Format results
        results = []
        for hits in search_results:
            for hit in hits:
                results.append({
                    "content": hit.entity.get("content"),
                    "relevance_score": float(hit.score),
                    "chunk_index": hit.entity.get("chunk_index")
                })
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in semantic search: {str(e)}")



@app.post("/search", response_model=SearchResponse)
async def search_website_content(request: SearchRequest):

    try:
        
        html_content = await fetch_html_content(request.url)
        
        clean_text = parse_html_content(html_content)
        
        chunks = tokenize_and_chunk_content(clean_text)
        
        index_chunks_to_milvus(chunks, request.url)
        
        search_results = search_similar_chunks(request.query, request.url, top_k=10)

        
        results = []
        for result in search_results:
            results.append(SearchResult(
                content=result["content"],
                relevance_score=result["relevance_score"],
                chunk_index=result["chunk_index"]
            ))
        
        return SearchResponse(
            results=results,
            total_chunks=len(chunks),
            query=request.query,
            url=request.url
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

