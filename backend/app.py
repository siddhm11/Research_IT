#!/usr/bin/env python3
# app.py - FastAPI backend for SPECTER2 search

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import logging

# Import your existing SPECTER2 search class
from specter2_search_enhanced import SPECTER2Search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the search system
search_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the search system on startup"""
    global search_system
    logger.info("🚀 Starting SPECTER2 Search System...")
    
    try:
        # Initialize search system (this takes time due to model loading)
        search_system = SPECTER2Search()
        logger.info("✅ SPECTER2 Search System initialized successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize search system: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down SPECTER2 Search System...")

# Initialize FastAPI app
app = FastAPI(
    title="SPECTER2 Research Search API",
    description="API for searching academic papers using SPECTER2 embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    top_k: int = Field(10, ge=1, le=50, description="Number of results to return")
    search_mode: str = Field("balanced", pattern="^(fast|balanced|quality)$", description="Search mode")
    fetch_metadata: bool = Field(True, description="Whether to fetch paper metadata")

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

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    mode_used: str
    total_results: int
    metadata_fetched: bool

class SmartSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    min_good_results: int = Field(3, ge=1, le=10)
    fetch_metadata: bool = Field(True)

class ComparisonRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)

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

# API Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "SPECTER2 Research Search API",
        "status": "healthy",
        "system_ready": search_system is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    return {
        "status": "healthy",
        "search_system": "ready",
        "gpu_available": search_system.gpu_count > 0,
        "gpu_count": search_system.gpu_count,
        "cached_papers": len(search_system.metadata_fetcher.cache)
    }

@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """Search papers using SPECTER2 embeddings"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        logger.info(f"🔍 Search request: '{request.query}' (mode: {request.search_mode})")
        
        # Perform search
        results, search_time, metadata_dict = search_system.search(
            query_text=request.query,
            top_k=request.top_k,
            search_mode=request.search_mode,
            fetch_metadata=request.fetch_metadata
        )
        
        # Format results
        formatted_results = []
        for result in results:
            arxiv_id = result.payload.get('arxiv_id', 'unknown')
            
            # Get metadata if available
            paper_metadata = None
            if request.fetch_metadata and metadata_dict:
                clean_id = search_system.metadata_fetcher._clean_arxiv_id(arxiv_id)
                raw_metadata = metadata_dict.get(clean_id)
                if raw_metadata:
                    paper_metadata = PaperMetadata(
                        arxiv_id=raw_metadata.arxiv_id,
                        title=raw_metadata.title,
                        authors=raw_metadata.authors,
                        abstract=raw_metadata.abstract,
                        published=raw_metadata.published,
                        categories=raw_metadata.categories,
                        doi=raw_metadata.doi,
                        journal_ref=raw_metadata.journal_ref
                    )
            
            formatted_results.append(SearchResult(
                score=result.score,
                arxiv_id=arxiv_id,
                metadata=paper_metadata
            ))
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            search_time_ms=search_time,
            mode_used=request.search_mode,
            total_results=len(results),
            metadata_fetched=request.fetch_metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/smart-search", response_model=SearchResponse)
async def smart_search_papers(request: SmartSearchRequest):
    """Smart search that automatically selects the best mode"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        logger.info(f"🧠 Smart search request: '{request.query}'")
        
        # Perform smart search
        results, search_time, mode_used, metadata_dict = search_system.smart_search(
            query_text=request.query,
            top_k=request.top_k,
            min_good_results=request.min_good_results,
            fetch_metadata=request.fetch_metadata
        )
        
        # Format results (same as regular search)
        formatted_results = []
        for result in results:
            arxiv_id = result.payload.get('arxiv_id', 'unknown')
            
            paper_metadata = None
            if request.fetch_metadata and metadata_dict:
                clean_id = search_system.metadata_fetcher._clean_arxiv_id(arxiv_id)
                raw_metadata = metadata_dict.get(clean_id)
                if raw_metadata:
                    paper_metadata = PaperMetadata(
                        arxiv_id=raw_metadata.arxiv_id,
                        title=raw_metadata.title,
                        authors=raw_metadata.authors,
                        abstract=raw_metadata.abstract,
                        published=raw_metadata.published,
                        categories=raw_metadata.categories,
                        doi=raw_metadata.doi,
                        journal_ref=raw_metadata.journal_ref
                    )
            
            formatted_results.append(SearchResult(
                score=result.score,
                arxiv_id=arxiv_id,
                metadata=paper_metadata
            ))
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            search_time_ms=search_time,
            mode_used=mode_used,
            total_results=len(results),
            metadata_fetched=request.fetch_metadata
        )
        
    except Exception as e:
        logger.error(f"❌ Smart search error: {e}")
        raise HTTPException(status_code=500, detail=f"Smart search failed: {str(e)}")

@app.post("/compare-modes", response_model=ComparisonResponse)
async def compare_search_modes(request: ComparisonRequest):
    """Compare all search modes for a query"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        logger.info(f"📊 Mode comparison request: '{request.query}'")
        
        # Perform comparison
        mode_results, best_mode, all_metadata = search_system.compare_search_modes(
            query_text=request.query,
            top_k=request.top_k,
            fetch_metadata=request.fetch_metadata
        )
        
        # Format comparison results
        comparisons = []
        for mode in ["fast", "balanced", "quality"]:
            mode_data = mode_results[mode]
            
            # Format results for this mode
            formatted_results = []
            for result in mode_data['results']:
                arxiv_id = result.payload.get('arxiv_id', 'unknown')
                
                paper_metadata = None
                if request.fetch_metadata and all_metadata:
                    clean_id = search_system.metadata_fetcher._clean_arxiv_id(arxiv_id)
                    raw_metadata = all_metadata.get(clean_id)
                    if raw_metadata:
                        paper_metadata = PaperMetadata(
                            arxiv_id=raw_metadata.arxiv_id,
                            title=raw_metadata.title,
                            authors=raw_metadata.authors,
                            abstract=raw_metadata.abstract,
                            published=raw_metadata.published,
                            categories=raw_metadata.categories,
                            doi=raw_metadata.doi,
                            journal_ref=raw_metadata.journal_ref
                        )
                
                formatted_results.append(SearchResult(
                    score=result.score,
                    arxiv_id=arxiv_id,
                    metadata=paper_metadata
                ))
            
            comparisons.append(ModeComparison(
                mode=mode,
                results=formatted_results,
                search_time_ms=mode_data['search_time'],
                high_confidence_count=mode_data['high_confidence'],
                good_results_count=mode_data['good_results'],
                average_score=mode_data['avg_score']
            ))
        
        return ComparisonResponse(
            query=request.query,
            comparisons=comparisons,
            recommended_mode=best_mode,
            total_metadata_fetched=len(all_metadata)
        )
        
    except Exception as e:
        logger.error(f"❌ Mode comparison error: {e}")
        raise HTTPException(status_code=500, detail=f"Mode comparison failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get search system statistics"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        # Get collection info safely
        collection_info = None
        try:
            collection_info = search_system.client.get_collection("arxiv_specter2_recommendations")
            points_count = collection_info.points_count
        except Exception:
            points_count = "unknown"
        
        return {
            "cached_papers": len(search_system.metadata_fetcher.cache),
            "gpu_count": search_system.gpu_count,
            "collection_name": "arxiv_specter2_recommendations",
            "collection_points": points_count,
            "model_loaded": search_system.model is not None,
            "tokenizer_loaded": search_system.tokenizer is not None,
            "qdrant_connected": search_system.client is not None
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/collection-info")
async def get_collection_info():
    """Get detailed Qdrant collection information"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        collection_info = search_system.client.get_collection("arxiv_specter2_recommendations")
        return {
            "collection_name": "arxiv_specter2_recommendations",
            "points_count": collection_info.points_count,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.name,
                "hnsw_config": {
                    "m": collection_info.config.params.hnsw_config.m,
                    "ef_construct": collection_info.config.params.hnsw_config.ef_construct,
                    "full_scan_threshold": collection_info.config.params.hnsw_config.full_scan_threshold,
                }
            },
            "status": collection_info.status.name
        }
    except Exception as e:
        logger.error(f"❌ Collection info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )