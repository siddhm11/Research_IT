#!/usr/bin/env python3
# app.py - FastAPI backend for SPECTER2 search

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import logging

# Assume your SPECTER2Search class is in a file named `specter.py`
from specter import SPECTER2Search

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
    description="An advanced API for semantic search of academic papers using SPECTER2, featuring multiple search modes.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
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

class SmartSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    min_good_results: int = Field(3, ge=1, le=10)
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False, description="Set to true to return the embedding vector for each result.")
    

class ComparisonRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
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

class ModeComparison(BaseModel):
    mode: str
    results: List[SearchResult]
    search_time_ms: float
    high_confidence_count: int
    good_results_count: int
    average_score: float

class ComparisonResponse(BaseModel):
    query: str
    comparisons: List[ModeComparison]
    recommended_mode: str
    total_metadata_fetched: int


# --- Helper Function ---

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
        
        # This is the new logic to handle the vector
        formatted.append(SearchResult(
            score=result.score,
            arxiv_id=arxiv_id,
            metadata=paper_metadata,
            vector=result.vector # <-- ADD THIS aSSIGNMENT
        ))
    return formatted

# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Provides a basic health check and welcome message."""
    return {"message": "SPECTER2 Research Search API", "status": "healthy", "system_ready": search_system is not None}

@app.get("/health", tags=["Health"])
async def health_check():
    """Provides a detailed health check of the search system components."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    return {
        "status": "healthy",
        "search_system": "ready",
        "gpu_available": search_system.gpu_count > 0,
        "gpu_count": search_system.gpu_count,
        "cached_papers": len(search_system.metadata_fetcher.cache)
    }

@app.post("/auto-search", response_model=SearchResponse, tags=["Search"])
async def auto_search_papers(request: AutoSearchRequest):
    """Automatically selects the best search strategy (title or semantic)."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    try:
        logger.info(f"🚀 Auto-search request: '{request.query}'")
        results, search_time, mode_used, metadata_dict = await asyncio.to_thread(
            search_system.auto_search,
            query_text=request.query, top_k=request.top_k,
            fetch_metadata=request.fetch_metadata, title_score_th=request.title_score_th,
            return_vector=request.return_vector # <-- ADD THIS PARAMETER
        )
        return SearchResponse(
            query=request.query, results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time, mode_used=mode_used,
            total_results=len(results), metadata_fetched=request.fetch_metadata
        )
    except Exception as e:
        logger.error(f"❌ Auto-search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_papers(request: SearchRequest):
    """Performs a search using a specific mode: fast, balanced, or quality."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    try:
        logger.info(f"🔍 Search request: '{request.query}' (mode: {request.search_mode})")
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_system.search,
            query_text=request.query, top_k=request.top_k,
            search_mode=request.search_mode, fetch_metadata=request.fetch_metadata
        )
        return SearchResponse(
            query=request.query, results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time, mode_used=request.search_mode,
            total_results=len(results), metadata_fetched=request.fetch_metadata
        )
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart-search", response_model=SearchResponse, tags=["Search"])
async def smart_search_papers(request: SmartSearchRequest):
    """Automatically escalates search mode to ensure a minimum number of good results."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    try:
        logger.info(f"🧠 Smart search request: '{request.query}'")
        results, search_time, mode_used, metadata_dict = await asyncio.to_thread(
            search_system.smart_search,
            query_text=request.query, top_k=request.top_k,
            min_good_results=request.min_good_results, fetch_metadata=request.fetch_metadata
        )
        return SearchResponse(
            query=request.query, results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time, mode_used=mode_used,
            total_results=len(results), metadata_fetched=request.fetch_metadata
        )
    except Exception as e:
        logger.error(f"❌ Smart search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare-modes", response_model=ComparisonResponse, tags=["Analysis"])
async def compare_search_modes(request: ComparisonRequest):
    """Compares performance and results across all search modes for a given query."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    try:
        logger.info(f"📊 Mode comparison request: '{request.query}'")
        mode_results, best_mode, all_metadata = await asyncio.to_thread(
            search_system.compare_search_modes,
            query_text=request.query, top_k=request.top_k,
            fetch_metadata=request.fetch_metadata
        )
        comparisons = []
        for mode, data in mode_results.items():
            comparisons.append(ModeComparison(
                mode=mode, results=format_results(data['results'], all_metadata, request.fetch_metadata),
                search_time_ms=data['search_time'], high_confidence_count=data['high_confidence'],
                good_results_count=data['good_results'], average_score=data['avg_score']
            ))
        return ComparisonResponse(
            query=request.query, comparisons=comparisons,
            recommended_mode=best_mode, total_metadata_fetched=len(all_metadata)
        )
    except Exception as e:
        logger.error(f"❌ Mode comparison error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", tags=["Diagnostics"])
async def get_stats():
    """Retrieves high-level statistics about the search system and its database."""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    try:
        points_count = "unknown"
        try:
            collection_info = await asyncio.to_thread(
                search_system.client.get_collection, collection_name=search_system.collection_name
            )
            points_count = collection_info.points_count
        except Exception as e:
            logger.warning(f"Could not retrieve collection point count: {e}")
        return {
            "cached_papers": len(search_system.metadata_fetcher.cache),
            "gpu_count": search_system.gpu_count,
            "main_collection_name": search_system.collection_name,
            "main_collection_points": points_count
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )