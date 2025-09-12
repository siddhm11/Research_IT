#!/usr/bin/env python3
# app.py - Complete Enhanced FastAPI backend with feed, personalization, and search
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple, Any, Union
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import logging
import time
import random
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models
# Import your modules
from minimal_specter import SPECTER2Search
from user_feed import UserEmbeddingManager, InteractionType, UserProfile
from user_types import VectorType
from user_mapping import get_or_create_uuid, get_uuid
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
search_system = None
user_manager = None



@asynccontextmanager
async def lifespan(app2: FastAPI):
    """Initialize systems on startup and clean up on shutdown."""
    global search_system, user_manager
    logger.info("🚀 Starting SPECTER2 Search System with User Storage..")
    try:
        # Initialize search system
        search_system = SPECTER2Search()
        logger.info("✅ SPECTER2 Search System initialized")
        
        # Initialize user management
        from pathlib import Path
        DB_PATH = "users.db"
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)   # directory safety
        user_manager = UserEmbeddingManager(db_path=DB_PATH)
        logger.info("✅ User Management System initialized")
        
        # in app.py lifespan():
        demo_user_count = len(user_manager.list_users())

        # Force skip demo creation
        logger.info(f"📊 Found {demo_user_count} users (skipping demo creation)")
        # if demo_user_count == 0:
        #     logger.info("🎭 Creating demo users...")
        #     await create_realistic_demo_users(user_manager, search_system)
        #     logger.info(f"✅ Demo users created successfully")

        
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize systems: {e}", exc_info=True)
        raise
    finally:
        logger.info("🔄 Shutting down systems...")
        search_system = None
        user_manager = None

# Initialize FastAPI app
app = FastAPI(
    title="SPECTER2 Research Search API with Feed & Personalization",
    description="A comprehensive API for semantic search of academic papers with personalized user profiles and feed generation",
    version="4.0.0",
    lifespan=lifespan
)

# Add CORS middleware
# From mini_app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://researchhub-three.vercel.app" , "http://localhost:3000", "http://localhost:3001", "http://localhost:8080", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Dependency Functions ---

def get_search_system() -> SPECTER2Search:
    """Dependency to get search system"""
    if search_system is None:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    return search_system

def get_user_manager() -> UserEmbeddingManager:
    """Dependency to get user manager"""
    if user_manager is None:
        raise HTTPException(status_code=503, detail="User management system not initialized")
    return user_manager

def get_user_profile(user_id: str, user_mgr: UserEmbeddingManager = Depends(get_user_manager)) -> UserProfile:
    """Dependency to get user profile"""
    user_profile = user_mgr.get_user(user_id)
    if user_profile is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user_profile

# --- Enhanced Pydantic Models ---

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    search_mode: str = Field("balanced", pattern="^(fast|balanced|quality)$")
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False)

class PersonalizedSearchRequest(BaseModel):
    user_id: str = Field(..., description="User ID for personalization")
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    search_mode: str = Field("balanced", pattern="^(fast|balanced|quality)$")
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False)
    complete_weight: float = Field(0.5, ge=0.0, le=1.0, description="Weight for complete user vector")
    subject_weight: float = Field(0.3, ge=0.0, le=1.0, description="Weight for subject-specific vector")

class AutoSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)
    title_score_th: float = Field(0.7, ge=0.5, le=1.0)
    return_vector: bool = Field(False)

class TitleSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False)

class SimilarityRequest(BaseModel):
    arxiv_id: str = Field(..., description="ArXiv ID of the paper")
    top_k: int = Field(10, ge=1, le=50)
    fetch_metadata: bool = Field(True)
    return_vector: bool = Field(False)

class CreateUserRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="Optional custom user ID")

class OnboardUserRequest(BaseModel):
    user_id: str = Field(..., description="User ID to onboard")
    subject1_name: str = Field(..., description="First research subject")
    subject2_name: str = Field(..., description="Second research subject")
    subject3_name: str = Field(..., description="Third research subject")
    subject_keywords: Optional[Dict[str, List[str]]] = Field(None, description="Keywords for each subject")
    liked_papers: Optional[List[str]] = Field(None, description="ArXiv IDs of papers user likes")

class AddInteractionRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    arxiv_id: str = Field(..., description="ArXiv ID of the paper")
    interaction_type: str = Field(..., pattern="^(like|dislike|view|bookmark)$")
    subject_area: Optional[str] = Field(None, description="Subject area of the paper")
    session_id: Optional[str] = Field(None, description="Session ID")

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

class FeedPaper(BaseModel):
    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published_date: str
    recommendation_score: float
    relevance_score: float
    diversity_score: float
    rank: int
    recommendation_reason: str
    stats: Dict[str, int]

class FeedResponse(BaseModel):
    user_id: str
    feed: List[FeedPaper]
    pagination: Dict[str, Any]
    user_context: Dict[str, Any]
    filters_applied: Dict[str, Any]

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    search_time_ms: float
    mode_used: str
    total_results: int
    metadata_fetched: bool
    personalized: bool = False
    user_id: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    is_onboarded: bool
    subjects: Dict[str, Dict]
    total_interactions: int
    created_at: str
    last_active: str

class UserListResponse(BaseModel):
    users: List[str]
    total_count: int

class TrendingResponse(BaseModel):
    papers: List[FeedPaper]
    count: int
    category_filter: Optional[str] = None

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

async def get_paper_vector(arxiv_id: str, search_sys: SPECTER2Search) -> Optional[np.ndarray]:
    """Helper to get paper vector from search system or sample data"""
    try:

        # Then try search system (your existing implementation)
        if search_sys.title_client is not None:
            results = await asyncio.to_thread(
                search_sys.title_client.scroll,
                collection_name="arxiv_papers_titles",
                scroll_filter={"must": [{"key": "arxiv_id", "match": {"value": arxiv_id}}]},
                limit=1,
                with_vectors=True
            )
            
            if results[0] and len(results[0]) > 0:
                return np.array(results[0][0].vector, dtype=np.float32)
        
        # Fallback to main collection
        results = await asyncio.to_thread(
            search_sys.client.scroll,
            collection_name=search_sys.collection_name,
            scroll_filter={"must": [{"key": "arxiv_id", "match": {"value": arxiv_id}}]},
            limit=1,
            with_vectors=True
        )
        
        if results[0] and len(results[0]) > 0:
            return np.array(results[0][0].vector, dtype=np.float32)
            
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting paper vector for {arxiv_id}: {e}")
        return None

def apply_mmr_to_papers(papers: List[Dict], user_profile: UserProfile,
                       query_vector: np.ndarray, lambda_param: float = 0.7,
                       max_results: int = 20) -> List[Dict]:
    """Apply MMR ranking to real papers from Qdrant"""
    if not papers:
        return []

    # Convert to vectors
    result_vectors = []
    for paper in papers:
        if paper.get('vector') is not None:
            result_vectors.append(paper['vector'])
        else:
            # Use dummy vector if no vector available
            result_vectors.append(np.random.normal(0, 0.3, 768))
    
    if not result_vectors:
        return []
    
    result_vectors = np.array(result_vectors)

    # MMR Algorithm
    selected_results = []
    remaining_indices = list(range(len(papers)))
    
    for rank in range(min(max_results, len(papers))):
        best_mmr_score = -float('inf')
        best_idx = None

        for idx in remaining_indices:
            # Relevance score
            relevance = np.dot(result_vectors[idx], query_vector)

            # Diversity score
            if selected_results:
                # Get vectors of already selected papers
                selected_vectors = [result_vectors[papers.index(next(p for p in papers if p['arxiv_id'] == selected_results[i]['arxiv_id']))] for i in range(len(selected_results))]
                if selected_vectors:
                    max_similarity = np.max([np.dot(result_vectors[idx], sv) for sv in selected_vectors])
                else:
                    max_similarity = 0.0
            else:
                max_similarity = 0.0

            # MMR Score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            paper = papers[best_idx]
            relevance_score = np.dot(result_vectors[best_idx], query_vector)
            diversity_score = 1 - (max_similarity if selected_results else 0)

            feed_paper = {
                'arxiv_id': paper['arxiv_id'],
                'title': paper['title'],
                'abstract': paper['abstract'],
                'authors': paper['authors'],
                'categories': paper['categories'],
                'published_date': paper['published_date'],
                'recommendation_score': float(best_mmr_score),
                'relevance_score': float(relevance_score),
                'diversity_score': float(diversity_score),
                'rank': rank,
                'recommendation_reason': 'HI',
                'stats': {
                    'views': random.randint(10, 500),
                    'bookmarks': random.randint(1, 50),
                    'likes': random.randint(0, 20)
                }
            }

            selected_results.append(feed_paper)
            remaining_indices.remove(best_idx)

    return selected_results
# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Provides a basic health check and welcome message."""
    return {
        "message": "SPECTER2 Research Search API with Feed & Personalization", 
        "status": "healthy", 
        "search_system_ready": search_system is not None,
        "user_system_ready": user_manager is not None,
        "version": "4.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Provides a detailed health check of all system components."""
    if search_system is None or user_manager is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
    title_search_available = (
        search_system.title_model is not None and 
        search_system.title_client is not None
    )
    
    return {
        "status": "healthy",
        "search_system": "ready",
        "user_system": "ready",
        "specter2_model": "loaded",
        "title_search_available": title_search_available,
        "cached_papers": len(search_system.metadata_fetcher.cache),
        "active_users": len(user_manager.users),
        "demo_users": len([u for u in user_manager.users.values() if u.is_onboarded])
    }

# --- Search Endpoints ---

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_papers(request: SearchRequest, search_sys: SPECTER2Search = Depends(get_search_system)):
    """Performs a SPECTER2 search using a specific mode: fast, balanced, or quality."""
    try:
        logger.info(f"🔍 SPECTER2 search request: '{request.query}' (mode: {request.search_mode})")
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_sys.search,
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
            metadata_fetched=request.fetch_metadata,
            personalized=False
        )
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/personalized-search", response_model=SearchResponse, tags=["Search"])
async def personalized_search(request: PersonalizedSearchRequest, 
                            search_sys: SPECTER2Search = Depends(get_search_system),
                            user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """Performs a personalized search based on user preferences."""
    try:
        # Get user profile
        user_profile = user_mgr.get_user(request.user_id)
        if user_profile is None:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        if not user_profile.is_onboarded:
            raise HTTPException(status_code=400, detail="User not onboarded. Please complete onboarding first.")
        
        logger.info(f"🎯 Personalized search for user {request.user_id}: '{request.query}'")
        
        # Generate base query vector
        base_query_vector = await asyncio.to_thread(
            search_sys._generate_query_embedding, 
            request.query
        )
        
        # Get personalized query vector
        personalized_vector = await asyncio.to_thread(
            user_profile.get_personalized_query_vector,
            base_query_vector,
            request.complete_weight,
            request.subject_weight
        )
        
        # Perform search with personalized vector
        start_time = time.time()
        ef_configs = {'fast': 32, 'balanced': 64, 'quality': 128}
        ef_search = ef_configs.get(request.search_mode, 64)
        
        results = await asyncio.to_thread(
            search_sys.client.search,
            collection_name=search_sys.collection_name,
            query_vector=personalized_vector.astype(np.float32).tolist(),
            limit=request.top_k,
            search_params=models.SearchParams(hnsw_ef=ef_search),
            with_payload=True,
            with_vectors=request.return_vector
        )
        
        search_time = (time.time() - start_time) * 1000
        
        # Fetch metadata if requested
        metadata_dict = {}
        if request.fetch_metadata and results:
            arxiv_ids = [r.payload.get('arxiv_id', '') for r in results if r.payload.get('arxiv_id')]
            if arxiv_ids:
                metadata_dict = await asyncio.to_thread(
                    search_sys.metadata_fetcher.fetch_batch_metadata, 
                    arxiv_ids
                )

        return SearchResponse(
            query=request.query,
            results=format_results(results, metadata_dict, request.fetch_metadata),
            search_time_ms=search_time,
            mode_used=f"personalized_{request.search_mode}",
            total_results=len(results),
            metadata_fetched=request.fetch_metadata,
            personalized=True,
            user_id=request.user_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Personalized search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto-search", response_model=SearchResponse, tags=["Search"])
async def auto_search_papers(request: AutoSearchRequest, search_sys: SPECTER2Search = Depends(get_search_system)):
    """Automatically selects the best search strategy (title or SPECTER2 semantic search)."""
    try:
        logger.info(f"🚀 Auto-search request: '{request.query}'")
        results, search_time, mode_used, metadata_dict = await asyncio.to_thread(
            search_sys.auto_search,
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
            metadata_fetched=request.fetch_metadata,
            personalized=False
        )
    except Exception as e:
        logger.error(f"❌ Auto-search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/title-search", response_model=SearchResponse, tags=["Search"])
async def title_search_papers(request: TitleSearchRequest, search_sys: SPECTER2Search = Depends(get_search_system)):
    """Performs a title-only search using MiniLM embeddings."""
    if search_sys.title_model is None or search_sys.title_client is None:
        raise HTTPException(status_code=503, detail="Title search components not available")
    
    try:
        logger.info(f"📚 Title search request: '{request.query}'")
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_sys.search_titles,
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
            metadata_fetched=request.fetch_metadata,
            personalized=False
        )
    except Exception as e:
        logger.error(f"❌ Title search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar", response_model=SearchResponse, tags=["Similarity"])
async def find_similar_papers(request: SimilarityRequest, search_sys: SPECTER2Search = Depends(get_search_system)):
    """Finds research papers that are semantically similar to a given ArXiv ID."""
    try:
        logger.info(f"🔎 Similarity request for ArXiv ID: '{request.arxiv_id}'")
        
        results, search_time, metadata_dict = await asyncio.to_thread(
            search_sys.find_similar_by_id,
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
            metadata_fetched=request.fetch_metadata,
            personalized=False
        )

    except ValueError as e:
        logger.warning(f"⚠️ Similarity search failed for ID '{request.arxiv_id}': {e}")
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"❌ Unexpected similarity search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal error occurred.")

# --- Feed Endpoints ---

async def get_real_papers_for_user_feed(
    user_profile: UserProfile, 
    search_sys: SPECTER2Search,
    count: int = 100
) -> List[Dict]:
    """Get real papers from Qdrant based on user preferences"""
    
    if not user_profile.is_onboarded:
        # For non-onboarded users, get recent papers
        try:
            # Get random sample of papers from Qdrant
            results = await asyncio.to_thread(
                search_sys.client.scroll,
                collection_name=search_sys.collection_name,
                limit=count,
                with_payload=True,
                with_vectors=True
            )
            
            papers = []
            if results[0]:  # results is tuple (points, next_page_offset)
                arxiv_ids = [r.payload.get('arxiv_id') for r in results[0] if r.payload.get('arxiv_id')]
                
                if arxiv_ids:
                    metadata_dict = await asyncio.to_thread(
                        search_sys.metadata_fetcher.fetch_batch_metadata,
                        arxiv_ids
                    )
                    
                    for point in results[0]:
                        arxiv_id = point.payload.get('arxiv_id')
                        if arxiv_id and arxiv_id in metadata_dict:
                            paper_metadata = metadata_dict[arxiv_id]
                            papers.append({
                                'arxiv_id': arxiv_id,
                                'title': paper_metadata.title,
                                'abstract': paper_metadata.abstract,
                                'authors': paper_metadata.authors,
                                'categories': paper_metadata.categories,
                                'published_date': paper_metadata.published,
                                'vector': np.array(point.vector) if point.vector else None,
                                'similarity_score': 0.8  # Default for trending
                            })
            return papers
        except Exception as e:
            logger.error(f"❌ Error getting trending papers: {e}")
            return []
    
    # For onboarded users - get personalized papers
    complete_vector = user_profile._retrieve_vector_from_qdrant(VectorType.COMPLETE)
    
    if complete_vector is None:
        logger.warning("⚠️ No complete vector found, falling back to trending")
        return await get_real_papers_for_user_feed(user_profile, search_sys, count)
    
    # Search for similar papers using user's preferences
    try:
        results = await asyncio.to_thread(
            search_sys.client.search,
            collection_name=search_sys.collection_name,
            query_vector=complete_vector.astype(np.float32).tolist(),
            limit=count,
            search_params=models.SearchParams(hnsw_ef=64),
            with_payload=True,
            with_vectors=True
        )
        
        # Convert to papers with metadata
        papers = []
        arxiv_ids = [r.payload.get('arxiv_id') for r in results if r.payload.get('arxiv_id')]
        
        if arxiv_ids:
            metadata_dict = await asyncio.to_thread(
                search_sys.metadata_fetcher.fetch_batch_metadata,
                arxiv_ids
            )
            
            for result in results:
                arxiv_id = result.payload.get('arxiv_id')
                if arxiv_id and arxiv_id in metadata_dict:
                    paper_metadata = metadata_dict[arxiv_id]
                    papers.append({
                        'arxiv_id': arxiv_id,
                        'title': paper_metadata.title,
                        'abstract': paper_metadata.abstract,
                        'authors': paper_metadata.authors,
                        'categories': paper_metadata.categories,
                        'published_date': paper_metadata.published,
                        'vector': np.array(result.vector) if result.vector else None,
                        'similarity_score': result.score
                    })
        
        return papers
        
    except Exception as e:
        logger.error(f"❌ Error getting personalized papers: {e}")
        return []

async def get_ranked_paper_ids(
    user_profile: UserProfile, 
    search_sys: SPECTER2Search,
    candidate_count: int = 100
) -> List[str]:
    """
    Performs a fast Qdrant search to get candidate vectors and then runs MMR
    to return a final, ranked list of ArXiv IDs. This function does NOT
    fetch any metadata from the external ArXiv API.
    """
    # 1. Get the user's "taste profile" vector
    user_query = user_profile._retrieve_vector_from_qdrant(VectorType.COMPLETE)
    if user_query is None:
        logger.warning(f"User {user_profile.user_id} has no query vector. Returning empty list.")
        return []

    # 2. Get 100 candidate papers (ID and Vector only) from Qdrant. This is fast.
    qdrant_results = await asyncio.to_thread(
        search_sys.client.search,
        collection_name=search_sys.collection_name,
        query_vector=user_query.astype(np.float32).tolist(),
        limit=candidate_count,
        with_payload=True,
        with_vectors=True  # Crucially, we get the vectors here
    )

    # 3. Prepare the list for the MMR algorithm.
    # The MMR function needs a list of dictionaries with specific keys.
    papers_for_mmr = [
        {
            # Required for MMR calculation:
            "arxiv_id": result.payload.get("arxiv_id"),
            "vector": np.array(result.vector),
            # Dummy data to match the function's expected input shape:
            "title": "", "abstract": "", "authors": [], "categories": [], "published_date": ""
        }
        for result in qdrant_results if result.payload.get("arxiv_id") and result.vector
    ]

    if not papers_for_mmr:
        return []

    # 4. Run MMR on the data with vectors to get the perfectly ranked list.
    ranked_papers = apply_mmr_to_papers(
        papers_for_mmr,
        user_profile,
        user_query,
        user_profile.mmr_lambda,
        max_results=len(papers_for_mmr)
    )

    # 5. Extract and return just the final, ranked list of ArXiv IDs.
    ranked_ids = [p['arxiv_id'] for p in ranked_papers]
    return ranked_ids

@app.get("/users/{user_id}/feed", response_model=FeedResponse, tags=["Feed"])
async def get_user_feed(
    user_id: str,
    count: int = Query(10, ge=1, le=50),
    page: int = Query(1, ge=1),
    user_profile: UserProfile = Depends(get_user_profile),
    search_sys: SPECTER2Search = Depends(get_search_system)
):
    """
    Get a personalized paper feed using the OPTIMIZED workflow:
    Rank first, then fetch metadata for only the required page.
    """
    try:
        # 1. FAST: Get the full list of ~100 paper IDs, already ranked by MMR.
        ranked_ids = await get_ranked_paper_ids(user_profile, search_sys)
        
        if not ranked_ids:
            return FeedResponse(user_id=user_id, feed=[], pagination={}, user_context={}, filters_applied={})

        # 2. Paginate the RANKED IDs to get just the 10 for the current page.
        start_idx = (page - 1) * count
        end_idx = start_idx + count
        paginated_ids = ranked_ids[start_idx:end_idx]

        if not paginated_ids:
            return FeedResponse(user_id=user_id, feed=[], pagination={}, user_context={}, filters_applied={"message": "Page number out of range"})

        # 3. SLOW BUT SMALL: Fetch metadata for ONLY the 10 paginated IDs.
        metadata_dict = await asyncio.to_thread(
            search_sys.metadata_fetcher.fetch_batch_metadata,
            paginated_ids
        )
        
        # 4. Build the final feed objects to send to the user.
        feed_papers = []
        for rank_in_page, arxiv_id in enumerate(paginated_ids):
            metadata = metadata_dict.get(arxiv_id)
            if metadata:
                feed_papers.append(FeedPaper(
                    arxiv_id=metadata.arxiv_id,
                    title=metadata.title,
                    abstract=metadata.abstract,
                    authors=metadata.authors,
                    categories=metadata.categories,
                    published_date=metadata.published,
                    # Scores are not passed through in this optimized flow, but the ranking is correct.
                    recommendation_score=0.0,
                    relevance_score=0.0,
                    diversity_score=0.0,
                    rank=start_idx + rank_in_page + 1,
                    recommendation_reason="Personalized for you",
                    stats={'views': 0, 'bookmarks': 0, 'likes': 0} # Placeholder stats
                ))

        # 5. Return the final, paginated response.
        return FeedResponse(
            user_id=user_id,
            feed=feed_papers,
            pagination={
                "page": page,
                "count": len(feed_papers),
                "requested_count": count,
                "has_more": len(ranked_ids) > end_idx
            },
            user_context={
                "is_onboarded": user_profile.is_onboarded,
                "mmr_lambda": user_profile.mmr_lambda
            },
            filters_applied={"using_optimized_flow": True}
        )

    except Exception as e:
        logger.error(f"❌ Error generating optimized feed for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))    

@app.get("/debug/user/{user_id}/vectors", tags=["Debug"])
async def debug_user_vectors(
    user_id: str,
    user_profile: UserProfile = Depends(get_user_profile),
    search_sys: SPECTER2Search = Depends(get_search_system)
):
    """Debug user vectors and search capability"""
    try:
        from user_feed import VectorType
        
        debug_info = {
            "user_id": user_id,
            "is_onboarded": user_profile.is_onboarded,
            "vectors": {},
            "search_test": {}
        }
        
        # Get all user vectors
        for vector_type in [VectorType.COMPLETE, VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
            vector = user_profile._retrieve_vector_from_qdrant(vector_type)
            debug_info["vectors"][vector_type.value] = {
                "exists": vector is not None,
                "shape": vector.shape if vector is not None else None,
                "norm": float(np.linalg.norm(vector)) if vector is not None else None
            }
        
        # Test search with complete vector
        complete_vector = user_profile._retrieve_vector_from_qdrant(VectorType.COMPLETE)
        if complete_vector is not None:
            try:
                test_results = await asyncio.to_thread(
                    search_sys.client.search,
                    collection_name=search_sys.collection_name,
                    query_vector=complete_vector.astype(np.float32).tolist(),
                    limit=10,
                    search_params=models.SearchParams(hnsw_ef=32),
                    with_payload=True,
                    with_vectors=False
                )
                
                debug_info["search_test"] = {
                    "results_found": len(test_results),
                    "top_scores": [r.score for r in test_results[:3]],
                    "arxiv_ids": [r.payload.get('arxiv_id') for r in test_results[:3]]
                }
            except Exception as e:
                debug_info["search_test"] = {"error": str(e)}
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def convert_search_results_to_papers(results, search_sys, limit):
    """Convert Qdrant search results to paper objects"""
    papers = []
    arxiv_ids = [r.payload.get('arxiv_id') for r in results[:limit*2] if r.payload.get('arxiv_id')]
    
    if arxiv_ids:
        metadata_dict = await asyncio.to_thread(
            search_sys.metadata_fetcher.fetch_batch_metadata,
            arxiv_ids
        )
        
        for result in results[:limit*2]:
            arxiv_id = result.payload.get('arxiv_id')
            if arxiv_id and arxiv_id in metadata_dict:
                paper_metadata = metadata_dict[arxiv_id]
                papers.append({
                    'arxiv_id': arxiv_id,
                    'title': paper_metadata.title,
                    'abstract': paper_metadata.abstract,
                    'authors': paper_metadata.authors,
                    'categories': paper_metadata.categories,
                    'published_date': paper_metadata.published,
                    'vector': np.array(result.vector) if result.vector else None,
                    'similarity_score': result.score,
                    'search_rank': len(papers)  # Track original search ranking
                })
                
                if len(papers) >= limit:
                    break
    
    return papers

def apply_feed_merge_strategy(feed_sections: Dict, strategy: str, target_size: int) -> List[Dict]:
    """Apply different strategies to merge feed sections"""
    
    if strategy == "separate":
        # Return sections as-is, organized by vector type
        return {
            "presentation": "sections",
            "sections": feed_sections
        }
    
    elif strategy == "weighted":
        # Merge based on vector strength and user engagement
        all_papers = []
        
        for section_name, section in feed_sections.items():
            vector_weight = 1.0  # Default weight
            
            if section_name == "complete_preferences":
                vector_weight = 0.6  # Complete vector gets high base weight
            elif "subject" in section_name:
                # Weight based on subject engagement
                subject_stats = section.get("subject_stats", {})
                interaction_count = subject_stats.get("interaction_count", 1)
                total_weight = subject_stats.get("total_weight", 0.1)
                vector_weight = 0.3 + (total_weight / 10.0)  # Subject weight based on learning
            
            # Add weighted papers
            for paper in section["papers"]:
                paper_copy = paper.copy()
                paper_copy["vector_source"] = section_name
                paper_copy["vector_weight"] = vector_weight
                paper_copy["weighted_score"] = paper["recommendation_score"] * vector_weight
                all_papers.append(paper_copy)
        
        # Sort by weighted score and deduplicate
        deduplicated_papers = deduplicate_papers(all_papers, key_field="weighted_score")
        sorted_papers = sorted(deduplicated_papers, key=lambda x: x["weighted_score"], reverse=True)
        
        return {
            "presentation": "unified",
            "papers": sorted_papers[:target_size],
            "merge_info": {
                "total_before_dedup": len(all_papers),
                "total_after_dedup": len(deduplicated_papers),
                "final_count": len(sorted_papers[:target_size])
            }
        }
    
    else:  # interleaved (default)
        # Interleave papers from different sections
        section_iterators = {}
        section_papers = {}
        
        for section_name, section in feed_sections.items():
            papers = section["papers"]
            section_papers[section_name] = papers
            section_iterators[section_name] = 0
        
        interleaved_papers = []
        rounds = 0
        max_rounds = target_size // len(feed_sections) + 2
        
        while len(interleaved_papers) < target_size and rounds < max_rounds:
            for section_name in feed_sections.keys():
                if (len(interleaved_papers) >= target_size or 
                    section_iterators[section_name] >= len(section_papers[section_name])):
                    continue
                
                paper = section_papers[section_name][section_iterators[section_name]].copy()
                paper["vector_source"] = section_name
                paper["interleave_round"] = rounds
                
                # Check for duplicates
                if not any(p["arxiv_id"] == paper["arxiv_id"] for p in interleaved_papers):
                    interleaved_papers.append(paper)
                
                section_iterators[section_name] += 1
            
            rounds += 1
        
        return {
            "presentation": "interleaved",
            "papers": interleaved_papers,
            "interleave_info": {
                "rounds_completed": rounds,
                "papers_per_section": {name: section_iterators[name] for name in feed_sections.keys()}
            }
        }

def deduplicate_papers(papers: List[Dict], key_field: str = "recommendation_score") -> List[Dict]:
    """Remove duplicate papers, keeping the one with highest score"""
    seen_ids = set()
    deduplicated = []
    
    # Sort by the key field to ensure we keep the best version
    sorted_papers = sorted(papers, key=lambda x: x.get(key_field, 0), reverse=True)
    
    for paper in sorted_papers:
        arxiv_id = paper["arxiv_id"]
        if arxiv_id not in seen_ids:
            seen_ids.add(arxiv_id)
            deduplicated.append(paper)
    
    return deduplicated

async def generate_cross_vector_analysis(feed_sections: Dict, user_profile: UserProfile, 
                                       search_sys: SPECTER2Search) -> Dict:
    """Analyze relationships and overlaps between different vector results"""
    analysis = {
        "vector_similarities": {},
        "paper_overlaps": {},
        "diversity_analysis": {},
        "recommendations": []
    }
    
    # 1. Calculate similarities between user vectors
    user_vectors = {}
    for vector_type in [VectorType.COMPLETE, VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
        vector = user_profile._retrieve_vector_from_qdrant(vector_type)
        if vector is not None:
            user_vectors[vector_type.value] = vector
    
    # Calculate pairwise similarities
    vector_names = list(user_vectors.keys())
    for i, vec1_name in enumerate(vector_names):
        for j, vec2_name in enumerate(vector_names):
            if i < j:
                vec1 = user_vectors[vec1_name]
                vec2 = user_vectors[vec2_name]
                similarity = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
                
                pair_key = f"{vec1_name}_vs_{vec2_name}"
                analysis["vector_similarities"][pair_key] = {
                    "similarity": similarity,
                    "interpretation": (
                        "High overlap" if similarity > 0.7 else
                        "Moderate overlap" if similarity > 0.4 else
                        "Distinct interests"
                    )
                }
    
    # 2. Analyze paper overlaps between sections
    section_papers = {}
    for section_name, section in feed_sections.items():
        paper_ids = {paper["arxiv_id"] for paper in section["papers"]}
        section_papers[section_name] = paper_ids
    
    section_names = list(section_papers.keys())
    for i, sect1 in enumerate(section_names):
        for j, sect2 in enumerate(section_names):
            if i < j:
                overlap = len(section_papers[sect1] & section_papers[sect2])
                total_unique = len(section_papers[sect1] | section_papers[sect2])
                
                overlap_key = f"{sect1}_vs_{sect2}"
                analysis["paper_overlaps"][overlap_key] = {
                    "overlap_count": overlap,
                    "overlap_percentage": overlap / max(len(section_papers[sect1]), 1) * 100,
                    "total_unique_papers": total_unique
                }
    
    # 3. Diversity analysis
    all_papers = []
    for section in feed_sections.values():
        all_papers.extend(section["papers"])
    
    unique_papers = len(set(paper["arxiv_id"] for paper in all_papers))
    total_papers = len(all_papers)
    
    analysis["diversity_analysis"] = {
        "total_papers_found": total_papers,
        "unique_papers": unique_papers,
        "duplication_rate": (total_papers - unique_papers) / max(total_papers, 1) * 100,
        "diversity_score": unique_papers / max(total_papers, 1) * 100
    }
    
    # 4. Generate recommendations
    if analysis["diversity_analysis"]["duplication_rate"] > 50:
        analysis["recommendations"].append("High overlap detected - consider diversifying research interests")
    
    if len([sim for sim in analysis["vector_similarities"].values() if sim["similarity"] > 0.8]) > 1:
        analysis["recommendations"].append("Some research areas are very similar - you might want to explore new fields")
    
    strongest_section = max(feed_sections.keys(), 
                          key=lambda x: len(feed_sections[x]["papers"]))
    analysis["recommendations"].append(f"Strongest interest area: {feed_sections[strongest_section]['title']}")
    
    return analysis



@app.get("/users/{user_id}/feed/multi-vector", response_model=Dict, tags=["Feed"])
async def get_multi_vector_feed(
    user_id: str,
    papers_per_vector: int = Query(15, ge=5, le=30),
    include_complete: bool = Query(True, description="Include complete vector results"),
    include_subjects: bool = Query(True, description="Include subject vector results"),
    merge_strategy: str = Query("interleaved", pattern="^(interleaved|separate|weighted)$"),
    user_profile: UserProfile = Depends(get_user_profile),
    search_sys: SPECTER2Search = Depends(get_search_system)
):
    """Get feed using multiple user vectors (complete + subjects) with flexible presentation"""
    try:
        if not user_profile.is_onboarded:
            raise HTTPException(status_code=400, detail="User must be onboarded for multi-vector feeds")
        
        feed_sections = {}
        vector_info = {}
        
        # 1. Get papers using COMPLETE vector
        if include_complete:
            complete_vector = user_profile._retrieve_vector_from_qdrant(VectorType.COMPLETE)
            if complete_vector is not None:
                logger.info(f"🎯 Searching with COMPLETE vector for {user_id}")
                
                complete_results = await asyncio.to_thread(
                    search_sys.client.search,
                    collection_name=search_sys.collection_name,
                    query_vector=complete_vector.astype(np.float32).tolist(),
                    limit=papers_per_vector * 2,  # Get more for filtering
                    search_params=models.SearchParams(hnsw_ef=64),
                    with_payload=True,
                    with_vectors=True
                )
                
                complete_papers = await convert_search_results_to_papers(
                    complete_results, search_sys, papers_per_vector
                )
                
                # Apply MMR to complete vector results
                complete_query = await asyncio.to_thread(
                    user_profile.get_personalized_query_vector_mmr,
                    complete_vector
                )
                
                feed_sections["complete_preferences"] = {
                    "title": "Overall Recommendations",
                    "description": "Papers matching your general research interests",
                    "vector_type": "complete",
                    "papers": apply_mmr_to_papers(
                        complete_papers, user_profile, complete_query, 
                        user_profile.mmr_lambda, papers_per_vector
                    )
                }
                
                vector_info["complete"] = {
                    "vector_norm": float(np.linalg.norm(complete_vector)),
                    "papers_found": len(complete_papers),
                    "search_quality": "high" if len(complete_papers) >= papers_per_vector else "partial"
                }
        
        # 2. Get papers using SUBJECT vectors
        if include_subjects:
            for vector_type in [VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
                subject = user_profile.subjects[vector_type]
                subject_vector = user_profile._retrieve_vector_from_qdrant(vector_type)
                
                if subject_vector is not None:
                    logger.info(f"🔬 Searching with {subject.name} vector")
                    
                    subject_results = await asyncio.to_thread(
                        search_sys.client.search,
                        collection_name=search_sys.collection_name,
                        query_vector=subject_vector.astype(np.float32).tolist(),
                        limit=papers_per_vector * 2,
                        search_params=models.SearchParams(hnsw_ef=64),
                        with_payload=True,
                        with_vectors=True
                    )
                    
                    subject_papers = await convert_search_results_to_papers(
                        subject_results, search_sys, papers_per_vector
                    )
                    
                    # Generate subject-specific personalized query
                    subject_query = await asyncio.to_thread(
                        user_profile.get_subject_personalized_query,
                        subject_vector, vector_type
                    )
                    
                    section_key = f"subject_{vector_type.value}"
                    feed_sections[section_key] = {
                        "title": f"{subject.name} Focus",
                        "description": f"Papers specifically related to {subject.name}",
                        "vector_type": vector_type.value,
                        "subject_name": subject.name,
                        "papers": apply_mmr_to_papers(
                            subject_papers, user_profile, subject_query,
                            user_profile.mmr_lambda, papers_per_vector
                        ),
                        "subject_stats": {
                            "interaction_count": subject.interaction_count,
                            "total_weight": subject.total_weight_accumulated,
                            "last_updated": subject.last_updated.isoformat(),
                            "keywords": subject.keywords
                        }
                    }
                    
                    vector_info[vector_type.value] = {
                        "vector_norm": float(np.linalg.norm(subject_vector)),
                        "papers_found": len(subject_papers),
                        "subject_strength": subject.total_weight_accumulated,
                        "search_quality": "high" if len(subject_papers) >= papers_per_vector else "partial"
                    }
        
        # 3. Apply merge strategy
        final_feed = apply_feed_merge_strategy(feed_sections, merge_strategy, papers_per_vector)
        
        # 4. Generate cross-vector analysis
        cross_vector_analysis = await generate_cross_vector_analysis(
            feed_sections, user_profile, search_sys
        )
        
        return {
            "user_id": user_id,
            "feed_type": "multi_vector",
            "merge_strategy": merge_strategy,
            "feed_sections": feed_sections,
            "merged_feed": final_feed,
            "vector_analysis": vector_info,
            "cross_vector_insights": cross_vector_analysis,
            "feed_metadata": {
                "papers_per_vector": papers_per_vector,
                "total_sections": len(feed_sections),
                "total_papers": sum(len(section["papers"]) for section in feed_sections.values()),
                "generation_time": datetime.now().isoformat(),
                "user_onboarded_at": user_profile.created_at.isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating multi-vector feed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- User Management Endpoints ---

@app.post("/users", tags=["Users"])
async def create_user(request: CreateUserRequest, user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """Create a new user profile."""
    try:
        user_profile = await asyncio.to_thread(user_mgr.create_user, request.user_id)
        
        return {
            "user_id": user_profile.user_id,
            "message": "User created successfully",
            "is_onboarded": user_profile.is_onboarded,
            "created_at": user_profile.created_at.isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ User creation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users", response_model=UserListResponse, tags=["Users"])
async def list_users(user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """List all users."""
    users = await asyncio.to_thread(user_mgr.list_users)
    return UserListResponse(users=users, total_count=len(users))

@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user(user_id: str, user_profile: UserProfile = Depends(get_user_profile)):
    """Get user profile information."""
    stats = await asyncio.to_thread(user_profile.get_stats)
    
    return UserResponse(
        user_id=stats["user_id"],
        is_onboarded=stats["is_onboarded"],
        subjects=stats["subjects"],
        total_interactions=stats["total_interactions"],
        created_at=stats["created_at"],
        last_active=stats["last_active"]
    )

@app.delete("/users/{user_id}", tags=["Users"])
async def delete_user(user_id: str, user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """Delete a user profile and all associated data."""
    try:
        await asyncio.to_thread(user_mgr.delete_user, user_id)
        return {"message": f"User {user_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"❌ User deletion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/onboard", tags=["Users"])
async def onboard_user(request: OnboardUserRequest, 
                      search_sys: SPECTER2Search = Depends(get_search_system),
                      user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """Onboard a user with research subjects and initial preferences."""
    try:
        # Get user profile
        user_profile = user_mgr.get_user(request.user_id)
        if user_profile is None:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        # Process liked papers if provided
        liked_papers_with_vectors = []
        if request.liked_papers:
            for arxiv_id in request.liked_papers:
                paper_vector = await get_paper_vector(arxiv_id, search_sys)
                if paper_vector is not None:
                    liked_papers_with_vectors.append((arxiv_id, paper_vector))
                else:
                    logger.warning(f"⚠️ Could not find vector for paper {arxiv_id}")
        
        # Onboard user
        await asyncio.to_thread(
            user_profile.onboard_user,
            request.subject1_name,
            request.subject2_name,
            request.subject3_name,
            liked_papers_with_vectors,
            request.subject_keywords
        )
        
        return {
            "message": f"User {request.user_id} onboarded successfully",
            "subjects": [request.subject1_name, request.subject2_name, request.subject3_name],
            "liked_papers_processed": len(liked_papers_with_vectors),
            "is_onboarded": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ User onboarding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/interactions", tags=["Users"])
async def add_interaction(request: AddInteractionRequest,
                         search_sys: SPECTER2Search = Depends(get_search_system),
                         user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """Add a user interaction with a paper."""
    try:
        # Get user profile
        user_profile = user_mgr.get_user(request.user_id)
        if user_profile is None:
            raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
        
        # Get paper vector
        paper_vector = await get_paper_vector(request.arxiv_id, search_sys)
        if paper_vector is None:
            raise HTTPException(status_code=404, detail=f"Paper {request.arxiv_id} not found")
        
        # Add interaction using the enhanced method with decay
        interaction_type = InteractionType(request.interaction_type)
        await asyncio.to_thread(
            user_profile.add_interaction_with_decay,
            request.arxiv_id,
            interaction_type,
            paper_vector,
            request.subject_area,
            request.session_id
        )
        
        return {
            "message": f"Interaction added successfully",
            "user_id": request.user_id,
            "arxiv_id": request.arxiv_id,
            "interaction_type": request.interaction_type,
            "subject_area": request.subject_area
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Add interaction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- Enhanced User Interaction Endpoints ---

@app.post("/users/{user_id}/interact", tags=["Users"])
async def record_interaction(
    user_id: str,
    arxiv_id: str = Query(..., description="ArXiv ID of the paper"),
    interaction_type: str = Query(..., description="Type of interaction (like, view, bookmark, dislike)"),
    subject_area: Optional[str] = Query(None, description="Subject area of the paper"),
    session_id: Optional[str] = Query(None, description="Session ID for tracking"),
    search_sys: SPECTER2Search = Depends(get_search_system),
    user_mgr: UserEmbeddingManager = Depends(get_user_manager)
):
    """Quick endpoint for recording user interactions (GET/POST compatible)"""
    try:
        # Get user profile
        user_profile = user_mgr.get_user(user_id)
        if user_profile is None:
            # Auto-create user if they don't exist
            user_profile = user_mgr.create_user(user_id)
            logger.info(f"🆕 Auto-created user {user_id}")
        
        # Get paper vector
        paper_vector = await get_paper_vector(arxiv_id, search_sys)
        if paper_vector is None:
            logger.warning(f"⚠️ Paper vector not found for {arxiv_id}, using dummy vector")
            paper_vector = np.random.normal(0, 0.3, 768).astype(np.float32)
        
        # Add interaction
        interaction_enum = InteractionType(interaction_type)
        await asyncio.to_thread(
            user_profile.add_interaction_with_decay,
            arxiv_id,
            interaction_enum,
            paper_vector,
            subject_area,
            session_id
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "arxiv_id": arxiv_id,
            "interaction_type": interaction_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Record interaction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/stats", tags=["Users"])
async def get_user_stats(user_id: str, user_profile: UserProfile = Depends(get_user_profile)):
    """Get detailed user statistics including decay analysis"""
    try:
        # Get basic stats
        basic_stats = await asyncio.to_thread(user_profile.get_stats)
        
        # Get decay statistics
        decay_stats = await asyncio.to_thread(user_profile.get_decay_statistics)
        
        return {
            "user_profile": basic_stats,
            "decay_analysis": decay_stats,
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Get user stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/mmr-config", tags=["Users"])
async def update_user_mmr_config(
    user_id: str,
    mmr_lambda: float = Query(..., ge=0.0, le=1.0, description="MMR lambda parameter (0=diversity, 1=relevance)"),
    user_profile: UserProfile = Depends(get_user_profile)
):
    """Update user's MMR configuration for feed diversity control"""
    try:
        await asyncio.to_thread(user_profile.update_mmr_lambda, mmr_lambda)
        
        return {
            "message": f"MMR lambda updated to {mmr_lambda:.2f}",
            "user_id": user_id,
            "relevance_weight": mmr_lambda,
            "diversity_weight": 1.0 - mmr_lambda,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Update MMR config error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# --- System Management & Statistics ---

@app.get("/stats", tags=["Diagnostics"])
async def get_stats():
    """Retrieves comprehensive system statistics."""
    if search_system is None or user_manager is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
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
        
        # Get user stats
        user_stats = await asyncio.to_thread(user_manager.get_all_stats)
        
        # Get decay summary
        decay_summary = await asyncio.to_thread(user_manager.get_decay_summary)
        
        return {
            "system": {
                "version": "4.0.0",
                "status": "healthy",
                "cached_papers": len(search_system.metadata_fetcher.cache),
            },
            "collections": {
                "main_collection_name": search_system.collection_name,
                "main_collection_points": main_points_count,
                "title_search_available": title_available,
                "title_collection_points": title_points_count
            },
            "users": user_stats,
            "decay_learning": decay_summary
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities", tags=["Diagnostics"])
async def get_capabilities():
    """Returns comprehensive information about available system capabilities."""
    if search_system is None or user_manager is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
    title_available = (
        search_system.title_model is not None and 
        search_system.title_client is not None
    )
    
    return {
        "search_capabilities": {
            "specter2_search": True,
            "title_search": title_available,
            "auto_search": True,
            "similarity_search": True,
            "personalized_search": True,
            "search_modes": ["fast", "balanced", "quality"]
        },
        "user_capabilities": {
            "user_management": True,
            "user_onboarding": True,
            "interaction_tracking": True,
            "decay_learning": True,
            "mmr_ranking": True,
            "subject_modeling": True
        },
        "feed_capabilities": {
            "personalized_feeds": True,
            "trending_papers": True,
            "category_filtering": True,
            "temporal_filtering": True,
            "pagination": True,
            "diversity_control": True
        },
        "technical_features": {
            "metadata_fetching": True,
            "vector_return": True,
            "batch_operations": True,
            "real_time_updates": True,
            "specter2_embeddings": True,
            "mmr_diversity": True,
            "temporal_decay": True
        },
        "available_endpoints": {
            "search": ["/search", "/personalized-search", "/auto-search", "/title-search", "/similar"],
            "users": ["/users", "/users/{user_id}", "/users/{user_id}/onboard", "/users/{user_id}/interactions", "/users/{user_id}/interact", "/users/{user_id}/stats", "/users/{user_id}/mmr-config"],
            "feed": ["/users/{user_id}/feed", "/papers/trending", "/papers/categories"],
            "system": ["/", "/health", "/stats", "/capabilities"]
        }
    }

@app.post("/system/cleanup", tags=["System"])
async def cleanup_system(
    days_threshold: int = Query(365, ge=30, le=730, description="Remove interactions older than N days"),
    user_mgr: UserEmbeddingManager = Depends(get_user_manager)
):
    """Cleanup old interactions across all users to optimize memory usage"""
    try:
        cleanup_stats = await asyncio.to_thread(user_mgr.cleanup_all_users, days_threshold)
        
        return {
            "message": "System cleanup completed successfully",
            "cleanup_stats": cleanup_stats,
            "days_threshold": days_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ System cleanup error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/batch-mmr-update", tags=["System"])
async def batch_update_mmr(
    mmr_lambda: float = Query(..., ge=0.0, le=1.0, description="New MMR lambda for all users"),
    user_mgr: UserEmbeddingManager = Depends(get_user_manager)
):
    """Update MMR lambda parameter for all users"""
    try:
        updated_count = await asyncio.to_thread(user_mgr.batch_update_mmr_lambda, mmr_lambda)
        
        return {
            "message": f"Updated MMR lambda to {mmr_lambda:.2f} for {updated_count} users",
            "updated_users": updated_count,
            "new_lambda": mmr_lambda,
            "relevance_weight": mmr_lambda,
            "diversity_weight": 1.0 - mmr_lambda,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Batch MMR update error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )