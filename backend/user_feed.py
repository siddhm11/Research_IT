#!/usr/bin/env python3
# Enhanced user_search.py - SPECTER2 with decay learning, selective updates, and MMR

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import math
import uuid
from enum import Enum
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
from minimal_db import SimpleUserDB

logger = logging.getLogger(__name__)


QDRANT_TITLE_URL = "https://6b25695f-de3c-4dbd-bb36-6de748ff47f2.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_TITLE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ug0KQAaAKM7Hv-L3NprJnvuLgNcNL9D9847dfWRL_Fk"
TITLE_COLLECTION_NAME = "arxiv_specter2_recommendations"
SPECTER2_MODEL_NAME = "allenai/specter2_base"  # Scientific paper embeddings
EMBEDDING_DIM = 768  

# Decay learning parameters
DECAY_RATE = 0.1  # Exponential decay rate (higher = faster decay)
MIN_WEIGHT_THRESHOLD = 0.05  # Minimum weight for very old interactions
MAX_INTERACTION_AGE_DAYS = 365  # Ignore interactions older than 1 year

class InteractionType(Enum):
    """Interaction types with base weights"""
    LIKE = "like"
    DISLIKE = "dislike" 
    VIEW = "view"
    BOOKMARK = "bookmark"

class VectorType(Enum):
    """Four user vector types"""
    COMPLETE = "complete"
    SUBJECT1 = "subject1"
    SUBJECT2 = "subject2"  
    SUBJECT3 = "subject3"

@dataclass
class Interaction:
    """User interaction with temporal decay support"""
    arxiv_id: str
    interaction_type: InteractionType
    timestamp: datetime
    subject_area: Optional[str] = None
    session_id: Optional[str] = None
    base_weight: float = field(init=False)  # Computed from interaction type
    
    def __post_init__(self):
        """Set base weight based on interaction type"""
        base_weights = {
            InteractionType.LIKE: 1.0,
            InteractionType.BOOKMARK: 0.8,
            InteractionType.VIEW: 0.3,
            InteractionType.DISLIKE: -0.5
        }
        self.base_weight = base_weights.get(self.interaction_type, 0.3)
    
    def age_days(self) -> float:
        """Get age of interaction in days"""
        return (datetime.now() - self.timestamp).total_seconds() / 86400
    
    def get_decayed_weight(self) -> float:
        """Calculate time-decayed weight for this interaction"""
        age = self.age_days()
        
        # Skip very old interactions
        if age > MAX_INTERACTION_AGE_DAYS:
            return 0.0
        
        # Exponential decay: weight = base_weight * exp(-decay_rate * age)
        decayed_weight = self.base_weight * math.exp(-DECAY_RATE * age)
        
        # Apply minimum threshold
        if abs(decayed_weight) < MIN_WEIGHT_THRESHOLD:
            return 0.0
            
        return decayed_weight

@dataclass
class UserSubject:
    """Represents one of the user's research subjects"""
    name: str
    vector_id: str
    keywords: List[str] = field(default_factory=list)
    interaction_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    total_weight_accumulated: float = 0.0  # Track total learning

@dataclass 
class MMRResult:
    """Result from MMR ranking"""
    arxiv_id: str
    relevance_score: float
    diversity_score: float
    mmr_score: float
    rank: int

class UserProfile:
    """Enhanced user profile with SPECTER2, decay learning, and MMR"""
    
    def __init__(self, user_id: str, db):
        self.user_id = user_id
        self.db = db
        self.embedding_dim = EMBEDDING_DIM
        # Store mapping for retrieval (you might want to persist this)

        # Qdrant client
        self.client = QdrantClient(
            url=QDRANT_TITLE_URL,
            api_key=QDRANT_TITLE_API_KEY,
            timeout=120
        )
        
        self.vector_ids=self._get_or_create_uuids()
        # SPECTER2 model for scientific papers
        try:
            self.specter_model = SentenceTransformer(SPECTER2_MODEL_NAME)
            logger.info(f"✅ Loaded SPECTER2 model")
        except Exception as e:
            logger.error(f"❌ Failed to load SPECTER2: {e}")
            # Fallback to SciBERT or MiniLM if SPECTER2 unavailable
            try:
                self.specter_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
                logger.info("🔄 Using SciBERT as fallback")
            except:
                self.specter_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                logger.info("🔄 Using MiniLM as final fallback")
        
        # User subjects with enhanced tracking
        self.subjects = {
            VectorType.SUBJECT1: UserSubject("Machine Learning", f"user_{user_id}_subject1"),
            VectorType.SUBJECT2: UserSubject("Computer Vision", f"user_{user_id}_subject2"), 
            VectorType.SUBJECT3: UserSubject("Natural Language Processing", f"user_{user_id}_subject3")
        }

        self.vector_id_mapping = {
            "complete": self.vector_ids[VectorType.COMPLETE],
            "subject1": self.vector_ids[VectorType.SUBJECT1], 
            "subject2": self.vector_ids[VectorType.SUBJECT2],
            "subject3": self.vector_ids[VectorType.SUBJECT3]
        }
        
        # Enhanced interaction tracking
        self.interactions: List[Interaction] = []
        self.mmr_lambda = 0.7  # Balance between relevance (0.7) and diversity (0.3)
        
        # User state
        self.is_onboarded = False
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        logger.info(f"🆕 Created enhanced user profile for {user_id}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate SPECTER2 embedding for scientific text"""
        try:
            # SPECTER2 works better with paper titles/abstracts
            vec = self.specter_model.encode(text, convert_to_numpy=True, device="cpu")
            return vec / (np.linalg.norm(vec) + 1e-8)  # Normalize
        except Exception as e:
            logger.error(f"❌ Error generating SPECTER2 embedding: {e}")
            raise
    
    def _store_vector_in_qdrant(self, vector_type: VectorType, vector: np.ndarray, metadata: Dict = None):
        """Store user vector with enhanced metadata"""
        try:
            vector_id = self.vector_ids[vector_type]
            
            if metadata is None:
                metadata = {}
            
            # Enhanced metadata tracking
            metadata.update({
                "user_id": self.user_id,
                "vector_type": vector_type.value,
                "is_user_vector": True,
                "model_type": "specter2",
                "last_updated": datetime.now().isoformat(),
                "update_count": metadata.get("update_count", 0) + 1
            })
            
            # Subject-specific metadata with learning stats
            if vector_type in self.subjects:
                subject = self.subjects[vector_type]
                metadata.update({
                    "subject_name": subject.name,
                    "subject_keywords": subject.keywords,
                    "interaction_count": subject.interaction_count,
                    "total_weight_accumulated": subject.total_weight_accumulated
                })
            
            # Store vector
            self.client.upsert(
                collection_name=TITLE_COLLECTION_NAME,
                points=[models.PointStruct(
                    id=vector_id,
                    vector=vector.astype(np.float32).tolist(),
                    payload=metadata
                )]
            )
            
            logger.info(f"✅ Updated {vector_type.value} vector (SPECTER2) for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store {vector_type.value} vector: {e}")
            raise
    
    def _get_or_create_uuids(self) -> Dict[VectorType, str]:
        """Get existing UUIDs or create new ones"""
        existing_uuids = self.db.get_user_uuids(self.user_id)
        
        if existing_uuids:
            print(f"✅ Restored UUIDs for {self.user_id}")
            return existing_uuids
        else:
            new_uuids = {
                VectorType.COMPLETE: str(uuid.uuid4()),
                VectorType.SUBJECT1: str(uuid.uuid4()),
                VectorType.SUBJECT2: str(uuid.uuid4()),
                VectorType.SUBJECT3: str(uuid.uuid4())
            }
            self.db.save_user_uuids(self.user_id, new_uuids)
            print(f"🆕 Created new UUIDs for {self.user_id}")
            return new_uuids
    
    
    def _retrieve_vector_from_qdrant(self, vector_type: VectorType) -> Optional[np.ndarray]:
        """Retrieve user vector from Qdrant"""
        try:
            vector_id = self.vector_ids[vector_type]
            
            result = self.client.retrieve(
                collection_name=TITLE_COLLECTION_NAME,
                ids=[vector_id],
                with_vectors=True
            )
            
            if result and len(result) > 0:
                vector = np.array(result[0].vector, dtype=np.float32)
                logger.info(f"📥 Retrieved {vector_type.value} vector for user {self.user_id}")
                return vector
            else:
                logger.info(f"🔍 No {vector_type.value} vector found for user {self.user_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to retrieve {vector_type.value} vector: {e}")
            return None
    
    def initialize_vectors(self):
        """Initialize user vectors in Qdrant with default embeddings"""
        try:
            # Initialize complete vector as zero vector
            complete_vector = np.zeros(self.embedding_dim, dtype=np.float32)
            self._store_vector_in_qdrant(VectorType.COMPLETE, complete_vector)
            
            # Initialize subject vectors from subject names using SPECTER2
            for vector_type, subject in self.subjects.items():
                subject_embedding = self._generate_embedding(subject.name)
                self._store_vector_in_qdrant(vector_type, subject_embedding)
            
            logger.info(f"🚀 Initialized all SPECTER2 vectors for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vectors: {e}")
            raise
    
    def add_interaction(self, arxiv_id: str, interaction_type: InteractionType, 
                       paper_vector: Optional[np.ndarray] = None,
                       subject_area: Optional[str] = None,
                       session_id: Optional[str] = None):
        """Add interaction and update relevant vectors (legacy method)"""
        
        interaction = Interaction(
            arxiv_id=arxiv_id,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            subject_area=subject_area,
            session_id=session_id
        )
        
        self.interactions.append(interaction)
        self.last_active = datetime.now()
        
        # Update vectors if paper vector is provided
        if paper_vector is not None:
            self._update_vectors_from_interaction(interaction, paper_vector)
        
        logger.info(f"📝 Added {interaction_type.value} interaction for paper {arxiv_id}")
    
    def add_interaction_with_decay(self, arxiv_id: str, interaction_type: InteractionType, 
                                 paper_vector: Optional[np.ndarray] = None,
                                 subject_area: Optional[str] = None,
                                 session_id: Optional[str] = None):
        """Add interaction with decay-aware vector updates (NEW METHOD)"""
        
        interaction = Interaction(
            arxiv_id=arxiv_id,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            subject_area=subject_area,
            session_id=session_id
        )
        
        self.interactions.append(interaction)
        self.last_active = datetime.now()
        
        # Update vectors with selective, decay-aware learning
        if paper_vector is not None:
            self._update_vectors_selective(interaction, paper_vector)
        
        logger.info(f"📝 Added {interaction_type.value} interaction with decay learning")
    
    def _update_vectors_from_interaction(self, interaction: Interaction, paper_vector: np.ndarray):
        """Legacy vector update method (updates all vectors)"""
        
        # Normalize paper vector
        paper_vector = paper_vector / (np.linalg.norm(paper_vector) + 1e-8)
        
        # Determine update weight based on interaction type
        weights = {
            InteractionType.LIKE: 1.0,
            InteractionType.BOOKMARK: 0.8,
            InteractionType.VIEW: 0.3,
            InteractionType.DISLIKE: -0.5
        }
        
        weight = weights.get(interaction.interaction_type, 0.3)
        
        # Update complete vector (always)
        self._update_single_vector(VectorType.COMPLETE, paper_vector, weight)
        
        # Update subject-specific vector if subject area is identified
        if interaction.subject_area:
            subject_vector_type = self._map_subject_to_vector_type(interaction.subject_area)
            if subject_vector_type:
                self._update_single_vector(subject_vector_type, paper_vector, weight)
                
                # Update subject interaction count
                self.subjects[subject_vector_type].interaction_count += 1
                self.subjects[subject_vector_type].last_updated = datetime.now()
    
    def _update_vectors_selective(self, interaction: Interaction, paper_vector: np.ndarray):
        """Selective vector updates - only update relevant subject + complete (NEW METHOD)"""
        
        # Normalize paper vector
        paper_vector = paper_vector / (np.linalg.norm(paper_vector) + 1e-8)
        
        # Get decayed weight
        current_weight = interaction.get_decayed_weight()
        if abs(current_weight) < MIN_WEIGHT_THRESHOLD:
            logger.info("⏰ Interaction too old/weak, skipping update")
            return
        
        # Always update complete vector (represents overall user preferences)
        self._update_single_vector_decay(VectorType.COMPLETE, paper_vector, current_weight)
        
        # Find most relevant subject vector (don't update all subjects!)
        relevant_subject = self._find_most_relevant_subject(paper_vector)
        
        if relevant_subject:
            self._update_single_vector_decay(relevant_subject, paper_vector, current_weight)
            
            # Update subject statistics
            self.subjects[relevant_subject].interaction_count += 1
            self.subjects[relevant_subject].total_weight_accumulated += abs(current_weight)
            self.subjects[relevant_subject].last_updated = datetime.now()
            
            logger.info(f"🎯 Updated {relevant_subject.value} subject vector (weight: {current_weight:.3f})")
        else:
            logger.info("🤷 No relevant subject found, only updated complete vector")
    
    def _find_most_relevant_subject(self, paper_vector: np.ndarray, 
                                  similarity_threshold: float = 0.3) -> Optional[VectorType]:
        """Find most relevant subject for paper using cosine similarity"""
        
        best_subject = None
        best_similarity = similarity_threshold  # Must exceed threshold
        
        for vector_type in [VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
            subject_vector = self._retrieve_vector_from_qdrant(vector_type)
            
            if subject_vector is not None:
                # Cosine similarity
                similarity = np.dot(paper_vector, subject_vector) / (
                    np.linalg.norm(paper_vector) * np.linalg.norm(subject_vector) + 1e-8
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_subject = vector_type
        
        return best_subject
    
    def _update_single_vector(self, vector_type: VectorType, new_vector: np.ndarray, weight: float):
        """Legacy vector update method"""
        
        # Retrieve current vector
        current_vector = self._retrieve_vector_from_qdrant(vector_type)
        
        if current_vector is None:
            # First interaction - use the new vector
            updated_vector = new_vector * abs(weight)
        else:
            # Incremental update with learning rate
            learning_rate = 0.1
            if weight < 0:  # Dislike - subtract from current vector
                updated_vector = current_vector - learning_rate * abs(weight) * new_vector
            else:  # Like/view - add to current vector
                updated_vector = current_vector + learning_rate * weight * new_vector
        
        # Normalize updated vector
        norm = np.linalg.norm(updated_vector)
        if norm > 1e-8:
            updated_vector = updated_vector / norm
        
        # Store back to Qdrant
        self._store_vector_in_qdrant(vector_type, updated_vector)
    
    def _update_single_vector_decay(self, vector_type: VectorType, new_vector: np.ndarray, 
                                  decayed_weight: float):
        """Update vector with decay-aware learning (NEW METHOD)"""
        
        # Retrieve current vector
        current_vector = self._retrieve_vector_from_qdrant(vector_type)
        
        if current_vector is None:
            # First interaction - initialize with weighted new vector
            updated_vector = new_vector * abs(decayed_weight)
        else:
            # Adaptive learning rate based on interaction recency and strength
            base_learning_rate = 0.1
            
            # Recent interactions get higher learning rate
            interaction_age = 0  # Current interaction age is 0
            recency_boost = math.exp(-DECAY_RATE * interaction_age)  # Always 1.0 for new interactions
            adaptive_lr = base_learning_rate * recency_boost
            
            if decayed_weight < 0:  # Negative feedback (dislike)
                # Subtract from current vector (push away)
                updated_vector = current_vector - adaptive_lr * abs(decayed_weight) * new_vector
            else:  # Positive feedback
                # Add to current vector (pull towards)
                updated_vector = current_vector + adaptive_lr * decayed_weight * new_vector
        
        # Normalize to maintain cosine similarity properties
        norm = np.linalg.norm(updated_vector)
        if norm > 1e-8:
            updated_vector = updated_vector / norm
        else:
            updated_vector = current_vector if current_vector is not None else new_vector
        
        # Store updated vector
        self._store_vector_in_qdrant(vector_type, updated_vector)
    
    def _map_subject_to_vector_type(self, subject_area: str) -> Optional[VectorType]:
        """Map a subject area string to vector type"""
        # Simple keyword matching - can be enhanced
        subject_area_lower = subject_area.lower()
        
        for vector_type, subject in self.subjects.items():
            subject_keywords = [subject.name.lower()] + [kw.lower() for kw in subject.keywords]
            if any(keyword in subject_area_lower for keyword in subject_keywords):
                return vector_type
        
        return None  # No matching subject found
    
    def onboard_user(self, subject1_name: str, subject2_name: str, subject3_name: str,
                    liked_papers: List[Tuple[str, np.ndarray]] = None,
                    subject_keywords: Dict[str, List[str]] = None):
        """
        Onboard user with custom subjects and initial preferences
        
        Args:
            subject1_name: Name of first research subject
            subject2_name: Name of second research subject  
            subject3_name: Name of third research subject
            liked_papers: List of (arxiv_id, paper_vector) for initial likes
            subject_keywords: Optional keywords for each subject
        """
        
        logger.info(f"🚀 Onboarding user {self.user_id}...")
        
        # Update subject names
        self.subjects[VectorType.SUBJECT1].name = subject1_name
        self.subjects[VectorType.SUBJECT2].name = subject2_name
        self.subjects[VectorType.SUBJECT3].name = subject3_name
        
        # Add keywords if provided
        if subject_keywords:
            for i, vector_type in enumerate([VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3], 1):
                subject_key = f"subject{i}"
                if subject_key in subject_keywords:
                    self.subjects[vector_type].keywords = subject_keywords[subject_key]
        
        # Initialize vectors
        self.initialize_vectors()
        
        # Process initial liked papers
        if liked_papers:
            for arxiv_id, paper_vector in liked_papers:
                # Try to determine subject area from paper
                subject_area = self._infer_subject_from_paper(paper_vector)
                
                # Use new decay-aware method
                self.add_interaction_with_decay(
                    arxiv_id=arxiv_id,
                    interaction_type=InteractionType.LIKE,
                    paper_vector=paper_vector,
                    subject_area=subject_area
                )
        
        self.is_onboarded = True
        
        logger.info(f"✅ User onboarded with subjects: {subject1_name}, {subject2_name}, {subject3_name}")
        logger.info(f"   📚 Initial papers: {len(liked_papers or [])}")
    
    def _infer_subject_from_paper(self, paper_vector: np.ndarray) -> Optional[str]:
        """Infer which subject a paper belongs to based on vector similarity"""
        
        best_subject = None
        best_similarity = -1.0
        
        for vector_type in [VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
            subject_vector = self._retrieve_vector_from_qdrant(vector_type)
            if subject_vector is not None:
                # Calculate cosine similarity
                similarity = np.dot(paper_vector, subject_vector) / (
                    np.linalg.norm(paper_vector) * np.linalg.norm(subject_vector) + 1e-8
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_subject = self.subjects[vector_type].name
        
        return best_subject if best_similarity > 0.3 else None  # Threshold for subject assignment
    
    def get_personalized_query_vector(self, original_query_vector: np.ndarray,
                                    complete_weight: float = 0.5,
                                    subject_weight: float = 0.3) -> np.ndarray:
        """Generate personalized query vector combining user preferences (LEGACY)"""
        
        if not self.is_onboarded:
            logger.info("👤 User not onboarded, returning original query")
            return original_query_vector
        
        # Normalize original query
        original_query_vector = original_query_vector / (np.linalg.norm(original_query_vector) + 1e-8)
        
        # Get complete user vector
        complete_vector = self._retrieve_vector_from_qdrant(VectorType.COMPLETE)
        if complete_vector is None:
            complete_vector = np.zeros_like(original_query_vector)
        
        # Get most relevant subject vector
        subject_vector = self._get_most_relevant_subject_vector(original_query_vector)
        if subject_vector is None:
            subject_vector = np.zeros_like(original_query_vector)
        
        # Combine vectors
        base_weight = 1.0 - complete_weight - subject_weight
        
        personalized_vector = (
            base_weight * original_query_vector +
            complete_weight * complete_vector +
            subject_weight * subject_vector
        )
        
        # Normalize
        norm = np.linalg.norm(personalized_vector)
        if norm > 1e-8:
            personalized_vector = personalized_vector / norm
        else:
            personalized_vector = original_query_vector
        
        logger.info(f"🎯 Generated personalized query (base: {base_weight:.2f}, complete: {complete_weight:.2f}, subject: {subject_weight:.2f})")
        
        return personalized_vector
    
    def get_personalized_query_vector_mmr(self, original_query_vector: np.ndarray,
                                         complete_weight: float = 0.5,
                                         subject_weight: float = 0.3) -> np.ndarray:
        """Generate personalized query vector for MMR-based search (NEW METHOD)"""
        
        if not self.is_onboarded:
            return original_query_vector
        
        # Normalize original query
        original_query_vector = original_query_vector / (np.linalg.norm(original_query_vector) + 1e-8)
        
        # Get complete user vector
        complete_vector = self._retrieve_vector_from_qdrant(VectorType.COMPLETE)
        if complete_vector is None:
            complete_vector = np.zeros_like(original_query_vector)
        
        # Get most relevant subject vector based on query
        subject_vector = self._get_most_relevant_subject_vector(original_query_vector)
        if subject_vector is None:
            subject_vector = np.zeros_like(original_query_vector)
        
        # Combine vectors with decay-weighted importance
        base_weight = 1.0 - complete_weight - subject_weight
        
        personalized_vector = (
            base_weight * original_query_vector +
            complete_weight * complete_vector +
            subject_weight * subject_vector
        )
        
        # Normalize final vector
        norm = np.linalg.norm(personalized_vector)
        if norm > 1e-8:
            personalized_vector = personalized_vector / norm
        else:
            personalized_vector = original_query_vector
        
        logger.info(f"🎯 Generated MMR-ready personalized query")
        return personalized_vector
    
    def apply_mmr_ranking(self, search_results: List[Dict], 
                         personalized_query: np.ndarray,
                         lambda_param: Optional[float] = None,
                         max_results: int = 20) -> List[MMRResult]:
        """Apply MMR (Maximal Marginal Relevance) for diverse results"""
        
        if lambda_param is None:
            lambda_param = self.mmr_lambda
        
        if not search_results:
            return []
        
        # Convert to numpy arrays for faster computation
        result_vectors = []
        result_metadata = []
        
        for result in search_results:
            if 'vector' in result:
                result_vectors.append(np.array(result['vector']))
                result_metadata.append(result)
        
        if not result_vectors:
            logger.warning("⚠️ No vectors found in search results for MMR")
            return []
        
        result_vectors = np.array(result_vectors)
        
        # MMR Algorithm
        selected_results = []
        remaining_indices = list(range(len(result_vectors)))
        
        for rank in range(min(max_results, len(result_vectors))):
            best_mmr_score = -float('inf')
            best_idx = None
            
            for idx in remaining_indices:
                # Relevance score (cosine similarity with personalized query)
                relevance = np.dot(result_vectors[idx], personalized_query)
                
                # Diversity score (max similarity with already selected results)
                if selected_results:
                    selected_vectors = result_vectors[[r.rank for r in selected_results]]
                    max_similarity = np.max([
                        np.dot(result_vectors[idx], selected_vec) 
                        for selected_vec in selected_vectors
                    ])
                else:
                    max_similarity = 0.0
                
                # MMR Score: λ * Relevance - (1-λ) * MaxSimilarity
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                # Add to selected results
                relevance_score = np.dot(result_vectors[best_idx], personalized_query)
                diversity_score = 1 - (max_similarity if selected_results else 0)
                
                mmr_result = MMRResult(
                    arxiv_id=result_metadata[best_idx].get('arxiv_id', f'paper_{best_idx}'),
                    relevance_score=float(relevance_score),
                    diversity_score=float(diversity_score),
                    mmr_score=float(best_mmr_score),
                    rank=rank
                )
                
                selected_results.append(mmr_result)
                remaining_indices.remove(best_idx)
        
        logger.info(f"🎲 Applied MMR ranking (λ={lambda_param:.2f}) to {len(selected_results)} results")
        return selected_results
    
    def _get_most_relevant_subject_vector(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        """Find most relevant subject vector for query"""
        
        best_vector = None
        best_similarity = -1.0
        
        for vector_type in [VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
            subject_vector = self._retrieve_vector_from_qdrant(vector_type)
            if subject_vector is not None:
                similarity = np.dot(query_vector, subject_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(subject_vector) + 1e-8
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_vector = subject_vector
        
        return best_vector
    
    def delete_user(self):
        """Delete all user vectors from Qdrant"""
        try:
            vector_ids = list(self.vector_ids.values())
            self.client.delete(
                collection_name=TITLE_COLLECTION_NAME,
                ids=vector_ids
            )
            logger.info(f"🗑️ Deleted all vectors for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete user vectors: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get user profile statistics"""
        return {
            "user_id": self.user_id,
            "is_onboarded": self.is_onboarded,
            "total_interactions": len(self.interactions),
            "subjects": {
                vector_type.value: {
                    "name": subject.name,
                    "keywords": subject.keywords,
                    "interaction_count": subject.interaction_count,
                    "last_updated": subject.last_updated.isoformat(),
                    "total_weight_accumulated": subject.total_weight_accumulated
                }
                for vector_type, subject in self.subjects.items()
            },
            "interaction_types": {
                itype.value: len([i for i in self.interactions if i.interaction_type == itype])
                for itype in InteractionType
            },
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "mmr_lambda": self.mmr_lambda,
            "model_type": "specter2"
        }
    
    def get_decay_statistics(self) -> Dict:
        """Get statistics about interaction decay and learning"""
        
        current_time = datetime.now()
        stats = {
            "total_interactions": len(self.interactions),
            "active_interactions": 0,  # Non-decayed interactions
            "total_current_weight": 0.0,
            "interaction_age_distribution": {"0-7days": 0, "7-30days": 0, "30-90days": 0, "90+days": 0},
            "subject_learning": {},
            "decay_parameters": {
                "decay_rate": DECAY_RATE,
                "min_weight_threshold": MIN_WEIGHT_THRESHOLD,
                "max_age_days": MAX_INTERACTION_AGE_DAYS
            }
        }
        
        # Analyze interaction decay
        for interaction in self.interactions:
            age = interaction.age_days()
            current_weight = interaction.get_decayed_weight()
            
            if abs(current_weight) >= MIN_WEIGHT_THRESHOLD:
                stats["active_interactions"] += 1
                stats["total_current_weight"] += abs(current_weight)
            
            # Age distribution
            if age <= 7:
                stats["interaction_age_distribution"]["0-7days"] += 1
            elif age <= 30:
                stats["interaction_age_distribution"]["7-30days"] += 1
            elif age <= 90:
                stats["interaction_age_distribution"]["30-90days"] += 1
            else:
                stats["interaction_age_distribution"]["90+days"] += 1
        
        # Subject learning statistics
        for vector_type, subject in self.subjects.items():
            stats["subject_learning"][vector_type.value] = {
                "name": subject.name,
                "interaction_count": subject.interaction_count,
                "total_weight_accumulated": subject.total_weight_accumulated,
                "last_updated": subject.last_updated.isoformat()
            }
        
        return stats
    
    def update_mmr_lambda(self, new_lambda: float):
        """Update MMR lambda parameter for diversity control"""
        if 0.0 <= new_lambda <= 1.0:
            old_lambda = self.mmr_lambda
            self.mmr_lambda = new_lambda
            logger.info(f"🎛️ Updated MMR lambda: {old_lambda:.2f} → {new_lambda:.2f}")
            logger.info(f"   Relevance: {new_lambda:.0%}, Diversity: {(1-new_lambda):.0%}")
        else:
            raise ValueError("MMR lambda must be between 0.0 and 1.0")
    
    def get_subject_similarities(self, query_vector: np.ndarray) -> Dict[str, float]:
        """Get similarities between query and all subject vectors"""
        similarities = {}
        
        for vector_type in [VectorType.SUBJECT1, VectorType.SUBJECT2, VectorType.SUBJECT3]:
            subject_vector = self._retrieve_vector_from_qdrant(vector_type)
            subject_name = self.subjects[vector_type].name
            
            if subject_vector is not None:
                similarity = np.dot(query_vector, subject_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(subject_vector) + 1e-8
                )
                similarities[subject_name] = float(similarity)
            else:
                similarities[subject_name] = 0.0
        
        return similarities
    
    def cleanup_old_interactions(self, days_threshold: int = MAX_INTERACTION_AGE_DAYS):
        """Remove interactions older than threshold to save memory"""
        initial_count = len(self.interactions)
        
        current_time = datetime.now()
        self.interactions = [
            interaction for interaction in self.interactions
            if interaction.age_days() <= days_threshold
        ]
        
        removed_count = initial_count - len(self.interactions)
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} old interactions (>{days_threshold} days)")
        
        return removed_count
    
    def get_subject_personalized_query(self, base_vector: np.ndarray, 
                                 vector_type: VectorType) -> np.ndarray:
        """Generate subject-specific personalized query combining subject and complete vectors"""
        subject_vector = self._retrieve_vector_from_qdrant(vector_type)
        complete_vector = self._retrieve_vector_from_qdrant(VectorType.COMPLETE)
        
        if subject_vector is None:
            return base_vector
        
        # Weight combination for subject-specific search
        subject_weight = 0.7  # High emphasis on specific subject
        complete_weight = 0.2  # Some general preference influence
        base_weight = 0.1     # Small influence from original query
        
        personalized = (
            base_weight * base_vector +
            subject_weight * subject_vector +
            complete_weight * (complete_vector if complete_vector is not None else np.zeros_like(base_vector))
        )
        
        # Normalize
        norm = np.linalg.norm(personalized)
        return personalized / norm if norm > 1e-8 else base_vector



class UserEmbeddingManager:
    """Manages multiple user profiles with enhanced features"""
    
    def __init__(self , db_path: str ="users.db"):
        self.users: Dict[str, UserProfile] = {}
        self.db = SimpleUserDB(db_path)
        logger.info("👥  UserEmbeddingManager with storage")
    
    def create_user(self, user_id: Optional[str] = None) -> UserProfile:
        """Create a new user profile"""
        
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        
        user_profile = UserProfile(user_id , self.db)
        self.users[user_id] = user_profile
        
        logger.info(f"👤 Created new user: {user_id}")
        return user_profile
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get existing user profile"""
        return self.users.get(user_id)
    
    def get_or_create_user(self, user_id: str) -> UserProfile:
        """Get existing user or create new one"""
        if user_id not in self.users:
            user_profile = UserProfile(user_id , self.db)
            self.users[user_id] = user_profile
            return user_profile
        return self.users[user_id]
    
    def delete_user(self, user_id: str):
        """Delete user profile and all vectors"""
        if user_id in self.users:
            self.users[user_id].delete_user()
            del self.users[user_id]
            logger.info(f"🗑️ Deleted user: {user_id}")
        else:
            logger.warning(f"⚠️ User {user_id} not found for deletion")
    
    def list_users(self) -> List[str]:
        """List all user IDs"""
        return list(self.users.keys())
    
    def get_onboarded_users(self) -> List[str]:
        """Get list of onboarded user IDs"""
        return [user_id for user_id, user in self.users.items() if user.is_onboarded]
    
    def get_all_stats(self) -> Dict:
        """Get comprehensive statistics for all users"""
        
        total_interactions = sum(len(user.interactions) for user in self.users.values())
        active_interactions = 0
        
        # Count active (non-decayed) interactions
        for user in self.users.values():
            for interaction in user.interactions:
                if abs(interaction.get_decayed_weight()) >= MIN_WEIGHT_THRESHOLD:
                    active_interactions += 1
        
        return {
            "total_users": len(self.users),
            "onboarded_users": len(self.get_onboarded_users()),
            "total_interactions": total_interactions,
            "active_interactions": active_interactions,
            "decay_efficiency": f"{(1 - active_interactions/max(total_interactions, 1)):.1%}",
            "model_type": "specter2",
            "embedding_dimension": EMBEDDING_DIM,
            "users": {user_id: user.get_stats() for user_id, user in self.users.items()}
        }
    
    def get_decay_summary(self) -> Dict:
        """Get summary of decay learning across all users"""
        
        summary = {
            "total_users": len(self.users),
            "total_interactions": 0,
            "active_interactions": 0,
            "subjects_with_learning": 0,
            "avg_weight_per_user": 0.0,
            "most_active_subjects": {},
            "interaction_age_summary": {"0-7days": 0, "7-30days": 0, "30-90days": 0, "90+days": 0}
        }
        
        total_weight = 0.0
        subject_weights = {}
        
        for user in self.users.values():
            decay_stats = user.get_decay_statistics()
            
            summary["total_interactions"] += decay_stats["total_interactions"]
            summary["active_interactions"] += decay_stats["active_interactions"]
            total_weight += decay_stats["total_current_weight"]
            
            # Aggregate age distribution
            for age_range, count in decay_stats["interaction_age_distribution"].items():
                summary["interaction_age_summary"][age_range] += count
            
            # Track subject learning
            for subject_name, subject_stats in decay_stats["subject_learning"].items():
                if subject_stats["total_weight_accumulated"] > 0:
                    summary["subjects_with_learning"] += 1
                    
                    subject_key = subject_stats["name"]
                    if subject_key not in subject_weights:
                        subject_weights[subject_key] = 0.0
                    subject_weights[subject_key] += subject_stats["total_weight_accumulated"]
        
        # Calculate averages
        if len(self.users) > 0:
            summary["avg_weight_per_user"] = total_weight / len(self.users)
        
        # Most active subjects
        summary["most_active_subjects"] = dict(
            sorted(subject_weights.items(), key=lambda x: x[1], reverse=True)[:10]
        )
        
        return summary
    
    def cleanup_all_users(self, days_threshold: int = MAX_INTERACTION_AGE_DAYS) -> Dict:
        """Cleanup old interactions for all users"""
        
        cleanup_stats = {
            "users_cleaned": 0,
            "total_removed": 0,
            "users_with_removals": []
        }
        
        for user_id, user in self.users.items():
            removed_count = user.cleanup_old_interactions(days_threshold)
            if removed_count > 0:
                cleanup_stats["users_cleaned"] += 1
                cleanup_stats["total_removed"] += removed_count
                cleanup_stats["users_with_removals"].append({
                    "user_id": user_id,
                    "removed_interactions": removed_count
                })
        
        logger.info(f"🧹 Cleanup complete: {cleanup_stats['total_removed']} interactions removed from {cleanup_stats['users_cleaned']} users")
        
        return cleanup_stats
    
    def batch_update_mmr_lambda(self, new_lambda: float) -> int:
        """Update MMR lambda for all users"""
        updated_count = 0
        
        for user in self.users.values():
            try:
                user.update_mmr_lambda(new_lambda)
                updated_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to update MMR lambda for user {user.user_id}: {e}")
        
        logger.info(f"🎛️ Updated MMR lambda to {new_lambda:.2f} for {updated_count} users")
        return updated_count
    
    def get_user_similarities_summary(self, query_vector: np.ndarray) -> Dict:
        """Get subject similarities for all users with a query"""
        
        similarities_summary = {}
        
        for user_id, user in self.users.items():
            if user.is_onboarded:
                try:
                    similarities = user.get_subject_similarities(query_vector)
                    similarities_summary[user_id] = {
                        "similarities": similarities,
                        "most_relevant": max(similarities.items(), key=lambda x: x[1]) if similarities else None
                    }
                except Exception as e:
                    logger.error(f"❌ Error computing similarities for user {user_id}: {e}")
                    similarities_summary[user_id] = {"error": str(e)}
        
        return similarities_summary
    
    


# Utility Functions

def create_sample_user_with_mmr(user_id: str, manager: UserEmbeddingManager) -> UserProfile:
    """Create a sample user with MMR configuration for testing"""
    
    user = manager.create_user(user_id)
    
    # Onboard with research subjects
    user.onboard_user(
        subject1_name="Deep Learning",
        subject2_name="Computer Vision", 
        subject3_name="Natural Language Processing",
        subject_keywords={
            "subject1": ["neural networks", "transformers", "attention", "backpropagation"],
            "subject2": ["image recognition", "convolutional networks", "object detection"],
            "subject3": ["language models", "sentiment analysis", "machine translation"]
        }
    )
    
    # Set MMR parameters for balanced exploration
    user.update_mmr_lambda(0.7)  # 70% relevance, 30% diversity
    
    logger.info(f"✅ Created sample user {user_id} with MMR configuration")
    return user

def analyze_mmr_performance(mmr_results: List[MMRResult]) -> Dict:
    """Analyze MMR ranking performance"""
    
    if not mmr_results:
        return {"error": "No MMR results to analyze"}
    
    relevance_scores = [r.relevance_score for r in mmr_results]
    diversity_scores = [r.diversity_score for r in mmr_results]
    mmr_scores = [r.mmr_score for r in mmr_results]
    
    analysis = {
        "total_results": len(mmr_results),
        "relevance": {
            "mean": np.mean(relevance_scores),
            "std": np.std(relevance_scores),
            "min": np.min(relevance_scores),
            "max": np.max(relevance_scores)
        },
        "diversity": {
            "mean": np.mean(diversity_scores),
            "std": np.std(diversity_scores),
            "min": np.min(diversity_scores),
            "max": np.max(diversity_scores)
        },
        "mmr_scores": {
            "mean": np.mean(mmr_scores),
            "std": np.std(mmr_scores),
            "min": np.min(mmr_scores),
            "max": np.max(mmr_scores)
        },
        "top_3_results": [
            {
                "rank": r.rank,
                "arxiv_id": r.arxiv_id,
                "relevance": r.relevance_score,
                "diversity": r.diversity_score,
                "mmr_score": r.mmr_score
            }
            for r in mmr_results[:3]
        ]
    }
    
    return analysis




# Example Usage and Testing

if __name__ == "__main__":
    # Initialize manager
    manager = UserEmbeddingManager()
    
    # Create sample user
    user = create_sample_user_with_mmr("test_user_123", manager)
    
    # Example query vector (normally from SPECTER2)
    query_vector = np.random.rand(EMBEDDING_DIM).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)
    
    # Get personalized query
    personalized_query = user.get_personalized_query_vector_mmr(query_vector)
    
    # Example search results (normally from Qdrant)
    sample_results = [
        {"arxiv_id": f"paper_{i}", "vector": np.random.rand(EMBEDDING_DIM), "score": 0.8 - i*0.05}
        for i in range(20)
    ]
    
    # Apply MMR ranking
    mmr_results = user.apply_mmr_ranking(sample_results, personalized_query, lambda_param=0.7)
    
    # Analyze performance
    mmr_analysis = analyze_mmr_performance(mmr_results)
    
    # Print results
    print("🎯 MMR Analysis Results:")
    print(f"   Total results: {mmr_analysis['total_results']}")
    print(f"   Avg relevance: {mmr_analysis['relevance']['mean']:.3f}")
    print(f"   Avg diversity: {mmr_analysis['diversity']['mean']:.3f}")
    print(f"   Top result: {mmr_analysis['top_3_results'][0]['arxiv_id']}")
    
    # Get decay statistics
    decay_stats = user.get_decay_statistics()
    print(f"\n⏰ Decay Learning Stats:")
    print(f"   Active interactions: {decay_stats['active_interactions']}/{decay_stats['total_interactions']}")
    print(f"   Total current weight: {decay_stats['total_current_weight']:.3f}")
    
    # Manager-level statistics
    manager_stats = manager.get_all_stats()
    print(f"\n👥 Manager Stats:")
    print(f"   Total users: {manager_stats['total_users']}")
    print(f"   Onboarded: {manager_stats['onboarded_users']}")
    print(f"   Model: {manager_stats['model_type']} ({manager_stats['embedding_dimension']}D)")
    
    logger.info("✅ Enhanced user search system demo completed successfully!")