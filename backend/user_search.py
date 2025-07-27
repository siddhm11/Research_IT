#!/usr/bin/env python3
# user_embeddings.py - Subject-based user profile management with Qdrant storage

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid

logger = logging.getLogger(__name__)

# Configuration - using the title Qdrant for user vectors
QDRANT_TITLE_URL = "https://ba0f9774-1b9e-4b0b-bb05-db8fadfe122c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_TITLE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sMVFQwd_dg3z89uIih5r5olFlbXLAjl_Gcx0V5IJG-U"
TITLE_COLLECTION_NAME = "arxiv_papers_titles"
MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

class InteractionType(Enum):
    """Simplified interaction types"""
    LIKE = "like"
    DISLIKE = "dislike" 
    VIEW = "view"
    BOOKMARK = "bookmark"

class VectorType(Enum):
    """Four user vector types"""
    COMPLETE = "complete"     # Overall user preferences
    SUBJECT1 = "subject1"     # First research subject
    SUBJECT2 = "subject2"     # Second research subject  
    SUBJECT3 = "subject3"     # Third research subject

@dataclass
class Interaction:
    """User interaction with a paper"""
    arxiv_id: str
    interaction_type: InteractionType
    timestamp: datetime
    subject_area: Optional[str] = None  # Which subject this paper belongs to
    session_id: Optional[str] = None
    
    def age_days(self) -> float:
        """Get age of interaction in days"""
        return (datetime.now() - self.timestamp).total_seconds() / 86400

@dataclass
class UserSubject:
    """Represents one of the user's research subjects"""
    name: str
    vector_id: str  # ID in Qdrant
    keywords: List[str] = field(default_factory=list)
    interaction_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

class UserProfile:
    """
    User profile with 4 vectors stored in Qdrant:
    1. Complete user vector (aggregated preferences)
    2-4. Three subject-specific vectors
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.embedding_dim = EMBEDDING_DIM
        
        # Qdrant client for storing user vectors
        self.client = QdrantClient(
            url=QDRANT_TITLE_URL,
            api_key=QDRANT_TITLE_API_KEY,
            timeout=120
        )
        
        # MiniLM model for generating embeddings
        self.title_model = SentenceTransformer(MINILM_MODEL_NAME)
        
        # User's three subjects (initialized as defaults, updated during onboarding)
        self.subjects = {
            VectorType.SUBJECT1: UserSubject("Machine Learning", f"user_{user_id}_subject1"),
            VectorType.SUBJECT2: UserSubject("Computer Vision", f"user_{user_id}_subject2"), 
            VectorType.SUBJECT3: UserSubject("Natural Language Processing", f"user_{user_id}_subject3")
        }
        
        # Vector IDs in Qdrant
        self.vector_ids = {
            VectorType.COMPLETE: f"user_{user_id}_complete",
            VectorType.SUBJECT1: f"user_{user_id}_subject1",
            VectorType.SUBJECT2: f"user_{user_id}_subject2",
            VectorType.SUBJECT3: f"user_{user_id}_subject3"
        }
        
        # Interaction history  
        self.interactions: List[Interaction] = []
        
        # User state
        self.is_onboarded = False
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
        logger.info(f"🆕 Created user profile for {user_id}")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate MiniLM embedding for text"""
        try:
            vec = self.title_model.encode(text, convert_to_numpy=True, device="cpu")
            return vec / np.linalg.norm(vec)  # Normalize
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            raise
    
    def _store_vector_in_qdrant(self, vector_type: VectorType, vector: np.ndarray, metadata: Dict = None):
        """Store a user vector in Qdrant"""
        try:
            vector_id = self.vector_ids[vector_type]
            
            # Default metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                "user_id": self.user_id,
                "vector_type": vector_type.value,
                "is_user_vector": True,
                "last_updated": datetime.now().isoformat()
            })
            
            # Add subject-specific metadata
            if vector_type in self.subjects:
                subject = self.subjects[vector_type]
                metadata.update({
                    "subject_name": subject.name,
                    "subject_keywords": subject.keywords,
                    "interaction_count": subject.interaction_count
                })
            
            # Upsert the vector
            self.client.upsert(
                collection_name=TITLE_COLLECTION_NAME,
                points=[models.PointStruct(
                    id=vector_id,
                    vector=vector.astype(np.float32).tolist(),
                    payload=metadata
                )]
            )
            
            logger.info(f"✅ Stored {vector_type.value} vector for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store vector {vector_type.value}: {e}")
            raise
    
    def _retrieve_vector_from_qdrant(self, vector_type: VectorType) -> Optional[np.ndarray]:
        """Retrieve a user vector from Qdrant"""
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
            logger.error(f"❌ Failed to retrieve vector {vector_type.value}: {e}")
            return None
    
    def initialize_vectors(self):
        """Initialize user vectors in Qdrant with default embeddings"""
        try:
            # Initialize complete vector as zero vector
            complete_vector = np.zeros(self.embedding_dim, dtype=np.float32)
            self._store_vector_in_qdrant(VectorType.COMPLETE, complete_vector)
            
            # Initialize subject vectors from subject names
            for vector_type, subject in self.subjects.items():
                subject_embedding = self._generate_embedding(subject.name)
                self._store_vector_in_qdrant(vector_type, subject_embedding)
            
            logger.info(f"🚀 Initialized all vectors for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize vectors: {e}")
            raise
    
    def add_interaction(self, arxiv_id: str, interaction_type: InteractionType, 
                       paper_vector: Optional[np.ndarray] = None,
                       subject_area: Optional[str] = None,
                       session_id: Optional[str] = None):
        """Add interaction and update relevant vectors"""
        
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
    
    def _update_vectors_from_interaction(self, interaction: Interaction, paper_vector: np.ndarray):
        """Update user vectors based on interaction"""
        
        # Normalize paper vector
        paper_vector = paper_vector / (np.linalg.norm(paper_vector) + 1e-8)
        
        # Determine update weight based on interaction type
        weights = {
            InteractionType.LIKE: 1.0,
            InteractionType.BOOKMARK: 0.8,
            InteractionType.VIEW: 0.3,
            InteractionType.DISLIKE: -0.5  # Negative weight
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
    
    def _update_single_vector(self, vector_type: VectorType, new_vector: np.ndarray, weight: float):
        """Update a single user vector with new information"""
        
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
                
                self.add_interaction(
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
        """Generate personalized query vector combining user preferences"""
        
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
    
    def _get_most_relevant_subject_vector(self, query_vector: np.ndarray) -> Optional[np.ndarray]:
        """Find the most relevant subject vector for the query"""
        
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
                    "last_updated": subject.last_updated.isoformat()
                }
                for vector_type, subject in self.subjects.items()
            },
            "interaction_types": {},
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
        }


class UserEmbeddingManager:
    """Manages multiple user profiles"""
    
    def __init__(self):
        self.users: Dict[str, UserProfile] = {}
        logger.info("👥 UserEmbeddingManager initialized")
    
    def create_user(self, user_id: Optional[str] = None) -> UserProfile:
        """Create a new user profile"""
        
        if user_id is None:
            user_id = str(uuid.uuid4())
        
        if user_id in self.users:
            raise ValueError(f"User {user_id} already exists")
        
        user_profile = UserProfile(user_id)
        self.users[user_id] = user_profile
        
        logger.info(f"👤 Created new user: {user_id}")
        return user_profile
    
    def get_user(self, user_id: str) -> Optional[UserProfile]:
        """Get existing user profile"""
        return self.users.get(user_id)
    
    def get_or_create_user(self, user_id: str) -> UserProfile:
        """Get existing user or create new one"""
        if user_id not in self.users:
            user_profile = UserProfile(user_id)
            self.users[user_id] = user_profile
            return user_profile
        return self.users[user_id]
    
    def delete_user(self, user_id: str):
        """Delete user profile and all vectors"""
        if user_id in self.users:
            self.users[user_id].delete_user()
            del self.users[user_id]
            logger.info(f"🗑️ Deleted user: {user_id}")
    
    def list_users(self) -> List[str]:
        """List all user IDs"""
        return list(self.users.keys())
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all users"""
        return {
            "total_users": len(self.users),
            "onboarded_users": sum(1 for user in self.users.values() if user.is_onboarded),
            "total_interactions": sum(len(user.interactions) for user in self.users.values()),
            "users": {user_id: user.get_stats() for user_id, user in self.users.items()}
        }