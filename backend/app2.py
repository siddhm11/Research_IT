#!/usr/bin/env python3
# app.py - Streamlined FastAPI backend for SPECTER2 search

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import logging

# Import your cleaned SPECTER2Search class
from minimal_specter import SPECTER2Search

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store the search system
search_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the search system on startup and clean up on shutdown."""
    global search_system
    logger.info("🚀 Starting SPECTER2 Search System...")
    try:
        search_system = SPECTER2Search()
        logger.info("✅ SPECTER2 Search System initialized successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize search system: {e}", exc_info=True)
        raise
    finally:
        logger.info("🔄 Shutting down SPECTER2 Search System...")
        search_system = None

# Initialize FastAPI app
app = FastAPI(
    title="SPECTER2 Research Search API",
    description="A streamlined API for semantic search of academic papers using SPECTER2",
    version="2.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    search_mode: str = Field("balanced", pattern="^(fast|balanced|quality)$")
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False, description="Set to true to return the embedding vector for each result.")

class AutoSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)
    title_score_th: float = Field(0.7, ge=0.5, le=1.0)
    return_vector: bool = Field(False, description="Set to true to return the embedding vector for each result.")

class TitleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Query text optimized for title search")
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False, description="Set to true to return the embedding vector for each result.")

class SimilarityRequest(BaseModel):
    arxiv_id: str = Field(..., description="The ArXiv ID of the paper to find similarities for.")
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False, description="Set to true to return the embedding vector for each result.")

class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    categories: List[str]
    doi: Optional[str] = None
    journal_ref: Optional[str] = None

class SearchResult(BaseModel):
    score: float
    arxiv_id: str
    metadata: Optional[PaperMetadata] = None
    vector: Optional[List[float]] = None 

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    mode_used: str
    total_results: int
    metadata_fetched: bool

# --- Helper Functions ---

def format_results(results_raw: list, metadata_dict: dict, fetch_metadata: bool) -> List[SearchResult]:
    """Helper to format raw search results into Pydantic models."""
    formatted = []
    if not results_raw or search_system is None:
        return formatted
        
    for result in results_raw:
        arxiv_id = result.payload.get('arxiv_id', 'unknown')
        paper_metadata = None
        
        if fetch_metadata and metadata_dict:
            clean_id = search_system.metadata_fetcher._clean_arxiv_id(arxiv_id)
            raw_metadata = metadata_dict.get(clean_id)
            if raw_metadata:
                paper_metadata = PaperMetadata(**raw_metadata.__dict__)
        
        formatted.append(SearchResult(
            score=result.score,
            arxiv_id=arxiv_id,
            metadata=paper_metadata,
            vector=getattr(result, 'vector', None) 
        ))
    return formatted

# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Provides a basic health check and welcome message."""
    return {
        "message": "SPECTER2 Research Search API", 
        "status": "healthy", 
        "system_ready": search_system is not None,
        "version": "2.1.0 (Streamlined)"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Provides a detailed health check of the search system components."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    # Check title search availability
    title_search_available = (
        search_system.title_model is not None and 
        search_system.title_client is not None
    )
    
    return {
        "status": "healthy",
        "search_system": "ready",
        "specter2_model": "loaded",
        "title_search_available": title_search_available,
        "cached_papers": len(search_system.metadata_fetcher.cache)
    }

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_papers(request: SearchRequest):
    """Performs a SPECTER2 search using a specific mode: fast, balanced, or quality."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        logger.info(f"🔍 SPECTER2 search request: '{request.query}' (mode: {request.search_mode})")
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_system.search,
            query_text=request.query, 
            top_k=request.top_k,
            search_mode=request.search_mode, 
            fetch_metadata=request.fetch_metadata,
            return_vector=request.return_vector
        )

        return SearchResponse(
            query=request.query, 
            results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time, 
            mode_used=request.search_mode,
            total_results=len(results), 
            metadata_fetched=request.fetch_metadata
        )
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-search", response_model=SearchResponse, tags=["Search"])
async def auto_search_papers(request: AutoSearchRequest):
    """Automatically selects the best search strategy (title or SPECTER2 semantic search)."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        logger.info(f"🚀 Auto-search request: '{request.query}'")
        results, search_time, mode_used, metadata_dict = await asyncio.to_thread(
            search_system.auto_search,
            query_text=request.query, 
            top_k=request.top_k,
            fetch_metadata=request.fetch_metadata, 
            title_score_th=request.title_score_th,
            return_vector=request.return_vector
        )
            
        return SearchResponse(
            query=request.query, 
            results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time, 
            mode_used=mode_used,
            total_results=len(results), 
            metadata_fetched=request.fetch_metadata
        )
    except Exception as e:
        logger.error(f"❌ Auto-search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/title-search", response_model=SearchResponse, tags=["Search"])
async def title_search_papers(request: TitleSearchRequest):
    """Performs a title-only search using MiniLM embeddings."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    if search_system.title_model is None or search_system.title_client is None:
        raise HTTPException(status_code=503, detail="Title search components not available")
    
    try:
        logger.info(f"📚 Title search request: '{request.query}'")
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_system.search_titles,
            query_text=request.query, 
            top_k=request.top_k,
            fetch_metadata=request.fetch_metadata,
            return_vector=request.return_vector
        )

        return SearchResponse(
            query=request.query, 
            results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time, 
            mode_used="title",
            total_results=len(results), 
            metadata_fetched=request.fetch_metadata
        )
    except Exception as e:
        logger.error(f"❌ Title search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar", response_model=SearchResponse, tags=["Similarity"])
async def find_similar_papers(request: SimilarityRequest):
    """Finds research papers that are semantically similar to a given ArXiv ID."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        logger.info(f"🔎 Similarity request for ArXiv ID: '{request.arxiv_id}'")
        
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_system.find_similar_by_id,
            arxiv_id=request.arxiv_id,
            top_k=request.top_k,
            fetch_metadata=request.fetch_metadata,
            return_vector=request.return_vector
        )
        
        return SearchResponse(
            query=f"Papers similar to {request.arxiv_id}",
            results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time,
            mode_used="similarity",
            total_results=len(results),
            metadata_fetched=request.fetch_metadata
        )

    except ValueError as e:
        logger.warning(f"⚠️ Similarity search failed for ID '{request.arxiv_id}': {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Unexpected similarity search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred.")

@app.get("/stats", tags=["Diagnostics"])
async def get_stats():
    """Retrieves high-level statistics about the search system and its databases."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        # Get main collection stats
        main_points_count = "unknown"
        try:
            collection_info = await asyncio.to_thread(
                search_system.client.get_collection, 
                collection_name=search_system.collection_name
            )
            main_points_count = collection_info.points_count
        except Exception as e:
            logger.warning(f"Could not retrieve main collection point count: {e}")
        
        # Get title collection stats
        title_points_count = "unknown"
        title_available = False
        try:
            if search_system.title_client is not None:
                title_collection_info = await asyncio.to_thread(
                    search_system.title_client.get_collection,
                    collection_name="arxiv_papers_titles"
                )
                title_points_count = title_collection_info.points_count
                title_available = True
        except Exception as e:
            logger.warning(f"Could not retrieve title collection point count: {e}")
        
        return {
            "cached_papers": len(search_system.metadata_fetcher.cache),
            "main_collection_name": search_system.collection_name,
            "main_collection_points": main_points_count,
            "title_search_available": title_available,
            "title_collection_points": title_points_count,
            "system_version": "2.1.0 (Streamlined)"
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities", tags=["Diagnostics"])
async def get_capabilities():
    """Returns information about available search capabilities."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    title_available = (
        search_system.title_model is not None and 
        search_system.title_client is not None
    )
    
    return {
        "search_modes": ["fast", "balanced", "quality"],
        "specter2_search": True,
        "title_search": title_available,
        "auto_search": True,
        "similarity_search": True,
        "metadata_fetching": True,
        "vector_return": True,
        "supported_endpoints": [
            "/search",
            "/auto-search", 
            "/title-search" if title_available else None,
            "/similar"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
    
    
