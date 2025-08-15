#!/usr/bin/env python3
"""
initialize_users.py - Corrected version

Syncs demo users into users.db and correctly initializes their user vectors 
inside the main Qdrant collection, following the application's logic.

Usage:
    python dummy_data.py
"""

import sys
import logging
import asyncio
import numpy as np
from typing import List, Dict, Optional

# Qdrant / embedding config
QDRANT_TITLE_URL = "https://6b25695f-de3c-4dbd-bb36-6de748ff47f2.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_TITLE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ug0KQAaAKM7Hv-L3NprJnvuLgNcNL9D9847dfWRL_Fk"
TITLE_COLLECTION_NAME = "arxiv_specter2_recommendations" # This is the main collection for papers AND users

# Demo user definitions
DEMO_PROFILES = [
    {
        "username": "alice_cv_expert",
        "field": "computer_vision",
        "subjects": ["Computer Vision", "Deep Learning", "Generative Models"],
    },
    {
        "username": "bob_nlp_researcher",
        "field": "natural_language_processing",
        "subjects": ["Natural Language Processing", "Large Language Models", "Transformers"],
    },
    {
        "username": "carol_robotics_engineer",
        "field": "robotics",
        "subjects": ["Robotics", "Reinforcement Learning", "Control Systems"],
    }
]

# ✅ ADDED MORE PAPERS for richer initial embeddings
CURATED_PAPERS_BY_FIELD = {
    "computer_vision": [
        {"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
        {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition"},
        {"arxiv_id": "2010.11929", "title": "An Image is Worth 16x16 Words"},
        {"arxiv_id": "1506.02640", "title": "You Only Look Once: YOLO"},
        {"arxiv_id": "1406.2661", "title": "Generative Adversarial Networks"},
        {"arxiv_id": "2104.07652", "title": "Masked Autoencoders Are Scalable Vision Learners"},
    ],
    "natural_language_processing": [
        {"arxiv_id": "1810.04805", "title": "BERT"},
        {"arxiv_id": "2005.14165", "title": "Language Models are Few-Shot Learners (GPT-3)"},
        {"arxiv_id": "1907.11692", "title": "RoBERTa"},
        {"arxiv_id": "1706.03762", "title": "Attention Is All You Need"}, # Also relevant here
        {"arxiv_id": "2302.13971", "title": "LLaMA: Open and Efficient Foundation Language Models"},
    ],
    "robotics": [
        {"arxiv_id": "1509.02971", "title": "Human-level control through deep reinforcement learning"},
        {"arxiv_id": "1707.06347", "title": "Proximal Policy Optimization Algorithms"},
        {"arxiv_id": "2006.11276", "title": "DDPG"},
        {"arxiv_id": "1606.01540", "title": "Asynchronous Methods for Deep Reinforcement Learning"},
    ],
}

# ========== Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("init-users")

# ========== Project Imports ==========
try:
    from qdrant_client import QdrantClient, models
    from user_feed import UserEmbeddingManager, InteractionType
except Exception as e:
    logger.error("Required project modules not found: %s", e)
    raise

# ========== Helpers ==========
def fetch_paper_vector_from_qdrant(client: QdrantClient, collection_name: str, arxiv_id: str) -> Optional[np.ndarray]:
    """Return the vector for a paper (if found) from the given collection_name."""
    try:
        res = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))]
            ),
            limit=1,
            with_vectors=True
        )
        if res and res[0] and len(res[0]) > 0:
            vec = res[0][0].vector
            if vec:
                return np.array(vec, dtype=np.float32)
        return None
    except Exception as e:
        logger.warning("Error fetching paper %s from Qdrant: %s", arxiv_id, e)
        return None

# ========== Main routine ==========
async def main():
    q_client = QdrantClient(url=QDRANT_TITLE_URL, api_key=QDRANT_TITLE_API_KEY)
    logger.info("Connected to Qdrant: %s", QDRANT_TITLE_URL)

    DB_PATH = r"C:/Users/siddh/_code_/Research_IT/backend/users.db"
    user_mgr = UserEmbeddingManager(db_path=DB_PATH)

    for profile in DEMO_PROFILES:
        username = profile["username"]
        field = profile["field"]
        subjects = profile["subjects"]

        # ✅ FIX 1: Use the username to create/get the user profile.
        # This ensures the API can find the user correctly.
        user_profile = user_mgr.get_or_create_user(username)
        logger.info(f"Found or created UserProfile for username: {username}")

        # Gather curated paper vectors for the user's field.
        curated = CURATED_PAPERS_BY_FIELD.get(field, [])
        liked_papers_with_vectors = []
        for paper in curated:
            arxiv_id = paper["arxiv_id"]
            vec = fetch_paper_vector_from_qdrant(q_client, TITLE_COLLECTION_NAME, arxiv_id)
            if vec is not None:
                liked_papers_with_vectors.append((arxiv_id, vec))
            else:
                logger.warning(f"Paper {arxiv_id} (field={field}) not found in collection '{TITLE_COLLECTION_NAME}'")

        logger.info(f"For user {username}, found {len(liked_papers_with_vectors)}/{len(curated)} curated paper vectors.")

        # Onboard the user. This method will correctly create and store ALL user vectors
        # in the right collection (`arxiv_specter2_recommendations`) internally.
        if liked_papers_with_vectors:
            s1, s2, s3 = subjects[0], subjects[1], subjects[2]
            try:
                user_profile.onboard_user(
                    subject1_name=s1,
                    subject2_name=s2,
                    subject3_name=s3,
                    liked_papers=liked_papers_with_vectors,
                    subject_keywords=None
                )
                logger.info(f"Onboarded user {username} with subjects [{s1}, {s2}, {s3}] and {len(liked_papers_with_vectors)} papers.")
            except Exception as e:
                logger.error(f"Failed to onboard user {username}: {e}")
                continue
        else:
            logger.warning(f"Skipping onboarding for {username} due to no found paper vectors.")

        # ✅ FIX 2: All manual vector calculation and upserting has been removed.
        # The `onboard_user` call above now handles all vector storage correctly.
        
        logger.info(f"✅ Finished initialization for user {username}.")

    logger.info("All demo users processed. Exiting.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        sys.exit(0)