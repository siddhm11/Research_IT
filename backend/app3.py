#!/usr/bin/env python3
# app.py - Enhanced FastAPI backend with user management and personalization
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
import asyncio
import uvicorn
from contextlib import asynccontextmanager
import logging
import time
from qdrant_client import models

# Import your modules
from minimal_specter import SPECTER2Search
from user_search import UserEmbeddingManager, InteractionType, UserProfile

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
    logger.info("🚀 Starting SPECTER2 Search System with User Management...")
    try:
        # Initialize search system
        search_system = SPECTER2Search()
        logger.info("✅ SPECTER2 Search System initialized")
        
        # Initialize user management
        user_manager = UserEmbeddingManager()
        logger.info("✅ User Management System initialized")
        
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
    title="SPECTER2 Research Search API with Personalization",
    description="A comprehensive API for semantic search of academic papers with personalized user profiles",
    version="3.0.0",
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

# --- Pydantic Models ---

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
    """Helper to get paper vector from search system"""
    try:
        # Try to get from title search first (faster)
        if search_sys.title_client is not None:
            results = await asyncio.to_thread(
                search_sys.title_client.scroll,
                collection_name="arxiv_papers_titles",
                scroll_filter={"must": [{"key": "arxiv_id", "match": {"value": arxiv_id}}]},
                limit=1,
                with_vectors=True
            )
            
            if results[0] and len(results[0]) > 0:
                import numpy as np
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
            import numpy as np
            return np.array(results[0][0].vector, dtype=np.float32)
            
        return None
        
    except Exception as e:
        logger.error(f"❌ Error getting paper vector for {arxiv_id}: {e}")
        return None

# --- API Endpoints ---

@app.get("/", tags=["Health"])
async def root():
    """Provides a basic health check and welcome message."""
    return {
        "message": "SPECTER2 Research Search API with Personalization", 
        "status": "healthy", 
        "search_system_ready": search_system is not None,
        "user_system_ready": user_manager is not None,
        "version": "3.0.0"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Provides a detailed health check of all system components."""
    if search_system is None or user_manager is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
    # Check title search availability
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
        "active_users": len(user_manager.users)
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
        
        import time
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
        
        # Add interaction
        interaction_type = InteractionType(request.interaction_type)
        await asyncio.to_thread(
            user_profile.add_interaction,
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

# --- Statistics and Diagnostics ---

@app.get("/stats", tags=["Diagnostics"])
async def get_stats():
    """Retrieves high-level statistics about the search system and users."""
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
        
        return {
            "cached_papers": len(search_system.metadata_fetcher.cache),
            "main_collection_name": search_system.collection_name,
            "main_collection_points": main_points_count,
            "title_search_available": title_available,
            "title_collection_points": title_points_count,
            "user_stats": user_stats,
            "system_version": "3.0.0"
        }
    except Exception as e:
        logger.error(f"❌ Stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities", tags=["Diagnostics"])
async def get_capabilities():
    """Returns information about available search capabilities."""
    if search_system is None or user_manager is None:
        raise HTTPException(status_code=503, detail="Systems not initialized")
    
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
        "personalized_search": True,
        "user_management": True,
        "metadata_fetching": True,
        "vector_return": True,
        "supported_endpoints": [
            "/search",
            "/personalized-search",
            "/auto-search", 
            "/title-search" if title_available else None,
            "/similar",
            "/users",
            "/users/{user_id}/onboard",
            "/users/{user_id}/interactions"
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