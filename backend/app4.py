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
import uuid
from datetime import datetime, timedelta
from qdrant_client import QdrantClient, models
# Import your modules
from minimal_specter import SPECTER2Search
from user_feed import UserEmbeddingManager, InteractionType, UserProfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
search_system = None
user_manager = None
sample_papers = []  # Will store sample papers for demo

# Sample paper generator for demo purposes
class SamplePaper:
    def __init__(self, arxiv_id: str, title: str, abstract: str, authors: List[str], 
                 categories: List[str], published_date: datetime, vector: np.ndarray):
        self.arxiv_id = arxiv_id
        self.title = title
        self.abstract = abstract
        self.authors = authors
        self.categories = categories
        self.published_date = published_date
        self.vector = vector
        self.view_count = random.randint(10, 500)
        self.bookmark_count = random.randint(1, 50)
        self.like_count = random.randint(0, 20)
        
        
def get_curated_papers_by_field() -> Dict[str, List[Dict]]:
    """Curated top papers from arXiv by research field"""
    
    return {
        "computer_vision": [
            {
                "arxiv_id": "1706.03762",  # Attention Is All You Need
                "title": "Attention Is All You Need",
                "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
                "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                "categories": ["cs.CV", "cs.LG"],
                "subjects": ["Computer Vision", "Deep Learning", "Attention Mechanisms"]
            },
            {
                "arxiv_id": "1512.03385",  # ResNet
                "title": "Deep Residual Learning for Image Recognition",
                "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously.",
                "authors": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
                "categories": ["cs.CV"],
                "subjects": ["Computer Vision", "Deep Learning", "Image Classification"]
            },
            {
                "arxiv_id": "2010.11929",  # Vision Transformer
                "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
                "abstract": "We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
                "authors": ["Alexey Dosovitskiy", "Lucas Beyer", "Alexander Kolesnikov"],
                "categories": ["cs.CV"],
                "subjects": ["Computer Vision", "Transformer Architecture", "Image Classification"]
            },
            {
                "arxiv_id": "1506.02640",  # YOLO
                "title": "You Only Look Once: Unified, Real-Time Object Detection",
                "abstract": "We present YOLO, a new approach to object detection that frames it as a regression problem to spatially separated bounding boxes.",
                "authors": ["Joseph Redmon", "Santosh Divvala", "Ross Girshick"],
                "categories": ["cs.CV"],
                "subjects": ["Computer Vision", "Object Detection", "Real-time Systems"]
            }
        ],
        
        "natural_language_processing": [
            {
                "arxiv_id": "1810.04805",  # BERT
                "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                "abstract": "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers.",
                "authors": ["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                "categories": ["cs.CL"],
                "subjects": ["Natural Language Processing", "Language Models", "Pre-training"]
            },
            {
                "arxiv_id": "2005.14165",  # GPT-3
                "title": "Language Models are Few-Shot Learners",
                "abstract": "We show that scaling up language models greatly improves task-agnostic, few-shot performance.",
                "authors": ["Tom B. Brown", "Benjamin Mann", "Nick Ryder"],
                "categories": ["cs.CL"],
                "subjects": ["Natural Language Processing", "Large Language Models", "Few-shot Learning"]
            },
            {
                "arxiv_id": "1907.11692",  # RoBERTa
                "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach",
                "abstract": "We present a replication study of BERT pretraining that carefully measures the impact of many key hyperparameters and training data size.",
                "authors": ["Yinhan Liu", "Myle Ott", "Naman Goyal"],
                "categories": ["cs.CL"],
                "subjects": ["Natural Language Processing", "Language Models", "Model Optimization"]
            }
        ],
        
        "machine_learning": [
            {
                "arxiv_id": "1412.6980",  # Adam Optimizer
                "title": "Adam: A Method for Stochastic Optimization",
                "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions.",
                "authors": ["Diederik P. Kingma", "Jimmy Ba"],
                "categories": ["cs.LG"],
                "subjects": ["Machine Learning", "Optimization", "Deep Learning"]
            },
            {
                "arxiv_id": "1406.2661",  # GANs
                "title": "Generative Adversarial Networks",
                "abstract": "We propose a new framework for estimating generative models via an adversarial process.",
                "authors": ["Ian J. Goodfellow", "Jean Pouget-Abadie", "Mehdi Mirza"],
                "categories": ["cs.LG"],
                "subjects": ["Machine Learning", "Generative Models", "Deep Learning"]
            }
        ],
        
        "robotics": [
            {
                "arxiv_id": "1509.02971",  # Deep Q-Network
                "title": "Human-level control through deep reinforcement learning",
                "abstract": "We develop a novel artificial agent that is capable of learning to play a diverse range of classic Atari 2600 video games.",
                "authors": ["Volodymyr Mnih", "Koray Kavukcuoglu", "David Silver"],
                "categories": ["cs.RO", "cs.LG"],
                "subjects": ["Robotics", "Reinforcement Learning", "Control Systems"]
            },
            {
                "arxiv_id": "1707.06347",  # PPO
                "title": "Proximal Policy Optimization Algorithms",
                "abstract": "We propose a new family of policy gradient methods for reinforcement learning.",
                "authors": ["John Schulman", "Filip Wolski", "Prafulla Dhariwal"],
                "categories": ["cs.RO", "cs.LG"],
                "subjects": ["Robotics", "Reinforcement Learning", "Policy Optimization"]
            }
        ]
    }

def check_papers_exist_in_qdrant(arxiv_ids: List[str], client: QdrantClient, 
                                collection_name: str) -> Dict[str, bool]:
    """Check which arXiv IDs exist in Qdrant collection with error handling"""
    
    existing_papers = {}
    
    for arxiv_id in arxiv_ids:
        try:
            # Use your payload index to efficiently search
            results = client.scroll(
                collection_name=collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="arxiv_id", 
                        match=models.MatchValue(value=arxiv_id)
                    )]
                ),
                limit=1,
                with_vectors=False  # Just checking existence
            )
            
            existing_papers[arxiv_id] = len(results[0]) > 0
            
            if existing_papers[arxiv_id]:
                logger.info(f"✅ Found paper {arxiv_id} in Qdrant")
            else:
                logger.warning(f"⚠️ Paper {arxiv_id} not found in Qdrant")
                
        except Exception as e:
            logger.error(f"❌ Error checking paper {arxiv_id}: {e}")
            existing_papers[arxiv_id] = False
    
    return existing_papers

def get_verified_papers_for_field(field_name: str, client: QdrantClient, 
                                 collection_name: str, max_papers: int = 5) -> List[Dict]:
    """Get verified papers that exist in Qdrant for a specific field"""
    
    curated_papers = get_curated_papers_by_field()
    
    if field_name not in curated_papers:
        logger.error(f"❌ Unknown field: {field_name}")
        return []
    
    field_papers = curated_papers[field_name]
    arxiv_ids = [paper["arxiv_id"] for paper in field_papers]
    
    # Check existence in Qdrant
    existing_papers = check_papers_exist_in_qdrant(arxiv_ids, client, collection_name)
    
    # Filter to only existing papers
    verified_papers = []
    for paper in field_papers:
        if existing_papers.get(paper["arxiv_id"], False):
            verified_papers.append(paper)
        else:
            logger.warning(f"⚠️ Skipping {paper['arxiv_id']}: {paper['title']}")
    
    # Limit to max_papers
    verified_papers = verified_papers[:max_papers]
    
    logger.info(f"📚 Found {len(verified_papers)}/{len(field_papers)} papers for {field_name}")
    return verified_papers

async def get_paper_vector_from_qdrant(arxiv_id: str, search_sys: SPECTER2Search) -> Optional[np.ndarray]:
    """Get paper vector directly from Qdrant using the payload index"""
    try:
        results = await asyncio.to_thread(
            search_sys.client.scroll,
            collection_name=search_sys.collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(
                    key="arxiv_id", 
                    match=models.MatchValue(value=arxiv_id)
                )]
            ),
            limit=1,
            with_vectors=True
        )
        
        if results[0] and len(results[0]) > 0:
            return np.array(results[0][0].vector, dtype=np.float32)
        else:
            logger.warning(f"⚠️ No vector found for {arxiv_id}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting vector for {arxiv_id}: {e}")
        return None

async def create_realistic_demo_users(user_mgr: UserEmbeddingManager, search_sys: SPECTER2Search):
    """Create demo users with real arXiv papers"""
    
    demo_profiles = [
        {
            "user_id": "alice_cv_expert",
            "field": "computer_vision",
            "subjects": ["Computer Vision", "Deep Learning", "Attention Mechanisms"],
            "expertise": "expert"
        },
        {
            "user_id": "bob_nlp_researcher", 
            "field": "natural_language_processing",
            "subjects": ["Natural Language Processing", "Language Models", "Pre-training"],
            "expertise": "advanced"
        },
        {
            "user_id": "carol_robotics_engineer",
            "field": "robotics", 
            "subjects": ["Robotics", "Reinforcement Learning", "Control Systems"],
            "expertise": "expert"
        }
    ]
    
    for profile in demo_profiles:
        # Create user
        user = user_mgr.create_user(profile["user_id"])
        
        # Get verified papers for this field
        verified_papers = get_verified_papers_for_field(
            profile["field"], 
            search_sys.client, 
            search_sys.collection_name,
            max_papers=5  # Reduced from 30 to 5!
        )
        
        if not verified_papers:
            logger.warning(f"⚠️ No verified papers found for {profile['field']}, skipping user {profile['user_id']}")
            continue
            
        # Onboard user
        try:
            user.onboard_user(
                subject1_name=profile["subjects"][0],
                subject2_name=profile["subjects"][1], 
                subject3_name=profile["subjects"][2]
            )
        except Exception as e:
            logger.error(f"❌ Failed to onboard {profile['user_id']}: {e}")
            continue
        
        # Add interactions with real papers
        for paper_info in verified_papers:
            try:
                # Get paper vector from Qdrant
                paper_vector = await get_paper_vector_from_qdrant(
                    paper_info["arxiv_id"], 
                    search_sys
                )
                
                if paper_vector is not None:
                    # Add realistic interaction
                    interaction_type = random.choices(
                        [InteractionType.LIKE, InteractionType.BOOKMARK, InteractionType.VIEW],
                        weights=[0.5, 0.3, 0.2]  # Experts tend to like/bookmark more
                    )[0]
                    
                    # Use actual subject from paper
                    subject_area = random.choice(paper_info["subjects"])
                    
                    user.add_interaction_with_decay(
                        arxiv_id=paper_info["arxiv_id"],
                        interaction_type=interaction_type,
                        paper_vector=paper_vector,
                        subject_area=subject_area
                    )
                    
                    logger.info(f"✅ Added {interaction_type.value} for {paper_info['arxiv_id']}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to add interaction for {paper_info['arxiv_id']}: {e}")
                continue
        
        logger.info(f"✅ Created user {profile['user_id']} with {len(verified_papers)} real papers")


@asynccontextmanager
async def lifespan(app2: FastAPI):
    """Initialize systems on startup and clean up on shutdown."""
    global search_system, user_manager, sample_papers
    logger.info("🚀 Starting SPECTER2 Search System with User Management & Feed...")
    try:
        # Initialize search system
        search_system = SPECTER2Search()
        logger.info("✅ SPECTER2 Search System initialized")
        
        # Initialize user management
        user_manager = UserEmbeddingManager()
        logger.info("✅ User Management System initialized")
        
        # Generate sample data for demo
        logger.info("🔄 Generating sample data...")
        await create_realistic_demo_users(user_manager, search_system)
        logger.info(f"✅ Generated {len(sample_papers)} sample papers and demo users")
        
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080", "http://localhost:5173"],
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
        # First check sample papers
        sample_paper = next((p for p in sample_papers if p.arxiv_id == arxiv_id), None)
        if sample_paper:
            return sample_paper.vector
        
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

def generate_recommendation_reason(user_profile: UserProfile, paper: SamplePaper) -> str:
    """Generate human-readable recommendation reason"""
    if not user_profile.is_onboarded:
        return "Trending in community"
    
    subject_names = [subject.name for subject in user_profile.subjects.values()]
    reasons = [
        f"Similar to papers you liked in {random.choice(subject_names)}",
        f"Popular among researchers with similar interests",
        f"Recent work in {paper.categories[0]}",
        f"Builds on concepts you've shown interest in",
        f"Highly cited work in your field"
    ]
    return random.choice(reasons)

def apply_mmr_to_sample_papers(papers: List[SamplePaper], user_profile: UserProfile, 
                              query_vector: np.ndarray, lambda_param: float = 0.7, 
                              max_results: int = 20) -> List[FeedPaper]:
    """Apply MMR ranking to sample papers"""
    if not papers:
        return []
    
    # Convert to vectors
    result_vectors = np.array([p.vector for p in papers])
    
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
                selected_indices = [i for i, r in enumerate(selected_results)]
                if selected_indices:
                    selected_vectors = np.array([result_vectors[papers.index(next(p for p in papers if p.arxiv_id == selected_results[i].arxiv_id))] for i in selected_indices])
                if len(selected_vectors) > 0:
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
            
            feed_paper = FeedPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                categories=paper.categories,
                published_date=paper.published_date.isoformat(),
                recommendation_score=float(best_mmr_score),
                relevance_score=float(relevance_score),
                diversity_score=float(diversity_score),
                rank=rank,
                recommendation_reason=generate_recommendation_reason(user_profile, paper),
                stats={
                    "views": paper.view_count,
                    "bookmarks": paper.bookmark_count,
                    "likes": paper.like_count
                }
            )
            
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
        "sample_papers_loaded": len(sample_papers),
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
        "sample_papers": len(sample_papers),
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

@app.get("/users/{user_id}/feed", response_model=FeedResponse, tags=["Feed"])
async def get_user_feed(
    user_id: str,
    count: int = Query(20, ge=1, le=50, description="Number of papers to return"),
    page: int = Query(1, ge=1, description="Page number"),
    category: Optional[str] = Query(None, description="Filter by category"),
    days_back: Optional[int] = Query(None, ge=1, le=365, description="Only papers from last N days"),
    user_profile: UserProfile = Depends(get_user_profile)
):
    """Get personalized paper feed for a user"""
    try:
        # Filter papers
        filtered_papers = sample_papers.copy()
        
        if category:
            filtered_papers = [p for p in filtered_papers if category in p.categories]
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_papers = [p for p in filtered_papers if p.published_date >= cutoff_date]
        
        if not user_profile.is_onboarded:
            # Return trending papers for non-onboarded users
            trending_papers = sorted(
                filtered_papers,
                key=lambda p: p.view_count + p.bookmark_count * 3 + p.like_count * 5,
                reverse=True
            )[:count]
            
            feed = [
                FeedPaper(
                    arxiv_id=p.arxiv_id,
                    title=p.title,
                    abstract=p.abstract,
                    authors=p.authors,
                    categories=p.categories,
                    published_date=p.published_date.isoformat(),
                    recommendation_score=0.8,
                    relevance_score=0.8,
                    diversity_score=0.5,
                    rank=i,
                    recommendation_reason="Trending in community",
                    stats={"views": p.view_count, "bookmarks": p.bookmark_count, "likes": p.like_count}
                )
                for i, p in enumerate(trending_papers)
            ]
            
            return FeedResponse(
                user_id=user_id,
                feed=feed,
                pagination={"page": page, "count": len(feed), "requested_count": count, "has_more": False},
                user_context={"is_onboarded": False, "total_interactions": 0, "subjects": [], "subject_similarities": {}, "mmr_lambda": 0.7},
                filters_applied={"category": category, "days_back": days_back, "total_papers_pool": len(filtered_papers)}
            )
        
        # Generate personalized query (simulate user's general interests)
        user_query = np.random.normal(0, 0.3, 768)
        user_query = user_query / (np.linalg.norm(user_query) + 1e-8)
        
        # Get personalized query vector
        personalized_query = await asyncio.to_thread(
            user_profile.get_personalized_query_vector_mmr,
            user_query
        )
        
        # Apply MMR to get diverse, personalized results
        feed_papers = apply_mmr_to_sample_papers(
            filtered_papers[:100],  # Top 100 candidates
            user_profile,
            personalized_query,
            user_profile.mmr_lambda,
            count
        )
        
        # Apply pagination
        start_idx = (page - 1) * count
        end_idx = start_idx + count
        paginated_feed = feed_papers[start_idx:end_idx]
        
        # User context
        user_stats = await asyncio.to_thread(user_profile.get_stats)
        subject_similarities = {}
        if paginated_feed:
            first_paper = next(p for p in sample_papers if p.arxiv_id == paginated_feed[0].arxiv_id)
            subject_similarities = await asyncio.to_thread(
                user_profile.get_subject_similarities,
                first_paper.vector
            )
        
        return FeedResponse(
            user_id=user_id,
            feed=paginated_feed,
            pagination={
                "page": page,
                "count": len(paginated_feed),
                "requested_count": count,
                "has_more": len(feed_papers) > end_idx
            },
            user_context={
                "is_onboarded": user_profile.is_onboarded,
                "total_interactions": user_stats["total_interactions"],
                "subjects": [subject["name"] for subject in user_stats["subjects"].values()],
                "subject_similarities": subject_similarities,
                "mmr_lambda": user_profile.mmr_lambda
            },
            filters_applied={
                "category": category,
                "days_back": days_back,
                "total_papers_pool": len(filtered_papers)
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating feed for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/trending", response_model=TrendingResponse, tags=["Feed"])
async def get_trending_papers(
    count: int = Query(20, ge=1, le=50),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """Get trending papers for discovery"""
    try:
        papers_pool = sample_papers
        if category:
            papers_pool = [p for p in papers_pool if category in p.categories]
        
        # Sort by engagement score
        trending_papers = sorted(
            papers_pool,
            key=lambda p: p.view_count + p.bookmark_count * 3 + p.like_count * 5,
            reverse=True
        )[:count]
        
        feed = [
            FeedPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                abstract=paper.abstract,
                authors=paper.authors,
                categories=paper.categories,
                published_date=paper.published_date.isoformat(),
                recommendation_score=0.8,
                relevance_score=0.8,
                diversity_score=0.5,
                rank=i,
                recommendation_reason="Trending in community",
                stats={
                    "views": paper.view_count,
                    "bookmarks": paper.bookmark_count,
                    "likes": paper.like_count
                }
            )
            for i, paper in enumerate(trending_papers)
        ]
        
        return TrendingResponse(papers=feed, count=len(feed), category_filter=category)
        
    except Exception as e:
        logger.error(f"❌ Error getting trending papers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/categories", tags=["Feed"])
async def get_categories():
    """Get available paper categories"""
    categories = {}
    for paper in sample_papers:
        for cat in paper.categories:
            categories[cat] = categories.get(cat, 0) + 1
    
    descriptions = {
        "cs.CV": "Computer Vision and Pattern Recognition",
        "cs.LG": "Machine Learning",
        "cs.CL": "Computational Linguistics and Natural Language Processing",
        "cs.RO": "Robotics"
    }
    
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "categories": [
            {
                "name": cat,
                "count": count,
                "description": descriptions.get(cat, cat)
            }
            for cat, count in sorted_categories
        ]
    }

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
                "sample_papers": len(sample_papers)
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

# --- Demo & Testing Endpoints ---

@app.get("/demo/papers", tags=["Demo"])
async def get_demo_papers(count: int = Query(10, ge=1, le=50)):
    """Get sample papers for demo purposes"""
    demo_papers = random.sample(sample_papers, min(count, len(sample_papers)))
    
    return {
        "papers": [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract[:200] + "..." if len(p.abstract) > 200 else p.abstract,
                "authors": p.authors,
                "categories": p.categories,
                "published_date": p.published_date.isoformat(),
                "stats": {
                    "views": p.view_count,
                    "bookmarks": p.bookmark_count,
                    "likes": p.like_count
                }
            }
            for p in demo_papers
        ],
        "total_available": len(sample_papers)
    }

@app.get("/demo/users", tags=["Demo"])
async def get_demo_users(user_mgr: UserEmbeddingManager = Depends(get_user_manager)):
    """Get demo user information"""
    demo_user_ids = ["alice_ml_researcher", "bob_nlp_student", "carol_robotics_engineer"]
    demo_users = []
    
    for user_id in demo_user_ids:
        user = user_mgr.get_user(user_id)
        if user:
            stats = await asyncio.to_thread(user.get_stats)
            demo_users.append({
                "user_id": user_id,
                "is_onboarded": stats["is_onboarded"],
                "subjects": [subject["name"] for subject in stats["subjects"].values()],
                "total_interactions": stats["total_interactions"],
                "mmr_lambda": stats.get("mmr_lambda", 0.7)
            })
    
    return {
        "demo_users": demo_users,
        "total_users": len(user_mgr.users)
    }

# --- Run the application ---

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )