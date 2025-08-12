#!/usr/bin/env python3
"""
initialize_users.py

Sync demo users into users1.db and into Qdrant as user vectors.

Usage:
    python initialize_users.py
"""

import sys
import logging
import asyncio
import numpy as np
from typing import List, Dict, Optional

# Qdrant / embedding config (from your message)
QDRANT_TITLE_URL = "https://6b25695f-de3c-4dbd-bb36-6de748ff47f2.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_TITLE_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ug0KQAaAKM7Hv-L3NprJnvuLgNcNL9D9847dfWRL_Fk"
TITLE_COLLECTION_NAME = "arxiv_specter2_recommendations"
SPECTER2_MODEL_NAME = "allenai/specter2_base"
EMBEDDING_DIM = 768

# Where to store user embeddings in Qdrant
USER_COLLECTION_NAME = "user_embeddings"   # created/used for user vectors

# Demo user definitions (you can extend or change these)
DEMO_PROFILES = [
    {
        "username": "alice_cv_expert",
        "field": "computer_vision",
        "subjects": ["Computer Vision", "Deep Learning", "Attention Mechanisms"],
    },
    {
        "username": "bob_nlp_researcher",
        "field": "natural_language_processing",
        "subjects": ["Natural Language Processing", "Language Models", "Pre-training"],
    },
    {
        "username": "carol_robotics_engineer",
        "field": "robotics",
        "subjects": ["Robotics", "Reinforcement Learning", "Control Systems"],
    }
]

# Curated sample papers (same structure you used in app.py)
CURATED_PAPERS_BY_FIELD = {
    "computer_vision": [
        {"arxiv_id": "1706.03762", "title": "Attention Is All You Need"},
        {"arxiv_id": "1512.03385", "title": "Deep Residual Learning for Image Recognition"},
        {"arxiv_id": "2010.11929", "title": "An Image is Worth 16x16 Words"},
        {"arxiv_id": "1506.02640", "title": "You Only Look Once: YOLO"},
    ],
    "natural_language_processing": [
        {"arxiv_id": "1810.04805", "title": "BERT"},
        {"arxiv_id": "2005.14165", "title": "Language Models are Few-Shot Learners (GPT-3)"},
        {"arxiv_id": "1907.11692", "title": "RoBERTa"},
    ],
    "machine_learning": [
        {"arxiv_id": "1412.6980", "title": "Adam"},
        {"arxiv_id": "1406.2661", "title": "GANs"},
    ],
    "robotics": [
        {"arxiv_id": "1509.02971", "title": "Human-level control through deep reinforcement learning"},
        {"arxiv_id": "1707.06347", "title": "PPO"},
    ],
}


# ========== Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("init-users")

# ========== Imports that depend on your repo ==========
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http import models as http_models  # sometimes needed depending on client
except Exception as e:
    logger.error("qdrant-client not available or failed to import: %s", e)
    raise

# These modules are from your project
try:
    # SPECTER2Search is used only if you need other helpers; for this script we only fetch vectors from Qdrant
    from minimal_specter import SPECTER2Search
    from user_feed import UserEmbeddingManager, InteractionType, UserProfile
    from user_mapping import get_or_create_uuid
except Exception as e:
    logger.error("Required project modules not found: %s", e)
    raise


# ========== Helpers ==========

def ensure_user_collection(client: QdrantClient, collection_name: str, dim: int = EMBEDDING_DIM):
    """
    Ensure the user collection exists in Qdrant with vector params. If it doesn't exist,
    try to create it. We'll avoid destructive recreate; if create fails we log and raise.
    """
    try:
        # try an upsert to see if collection exists
        # If collection doesn't exist, attempt to create it.
        # Some qdrant client versions have get_collections or get_collection; using create_collection is enough.
        client.get_collection(collection_name)
        logger.info("Qdrant collection '%s' exists.", collection_name)
    except Exception:
        logger.info("Qdrant collection '%s' not found - creating with dim=%d", collection_name, dim)
        try:
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            logger.info("Created collection '%s' in Qdrant", collection_name)
        except Exception as e:
            logger.error("Failed to create collection '%s': %s", collection_name, e)
            raise


def fetch_paper_vector_from_qdrant(client: QdrantClient, collection_name: str, arxiv_id: str) -> Optional[np.ndarray]:
    """Return the vector for a paper (if found) from the given collection_name."""
    try:
        # Use payload index to find the arxiv_id
        res = client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="arxiv_id",
                        match=models.MatchValue(value=arxiv_id)
                    )
                ]
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


def upsert_user_vector_to_qdrant(client: QdrantClient, collection_name: str, user_id: str, vector: np.ndarray):
    """Upsert a single user vector into Qdrant (point id = user_id)."""
    try:
        point = models.PointStruct(
            id=user_id,
            vector=vector.astype(np.float32).tolist(),
            payload={"user_id": user_id, "entity": "user"}
        )
        client.upsert(collection_name=collection_name, points=[point])
        logger.info("Upserted user vector for %s into Qdrant collection '%s'", user_id, collection_name)
    except Exception as e:
        logger.error("Failed to upsert user vector for %s: %s", user_id, e)
        raise


# ========== Main routine ==========
async def main():
    # Init Qdrant client (title/paper collection)
    q_client = QdrantClient(url=QDRANT_TITLE_URL, api_key=QDRANT_TITLE_API_KEY)
    logger.info("Connected to Qdrant (title): %s", QDRANT_TITLE_URL)

    # Ensure user collection exists (create if missing)
    try:
        ensure_user_collection(q_client, USER_COLLECTION_NAME, EMBEDDING_DIM)
    except Exception as e:
        logger.error("Could not ensure user collection in Qdrant: %s", e)
        return

    # Init local user manager (SQLite)
    DB_PATH = r"C:/Users/siddh/_code_/Research_IT/backend/users.db"
    user_mgr = UserEmbeddingManager(db_path=DB_PATH)

    # If you have SPECTER2Search and want to use it for any extra steps:
    # specter = SPECTER2Search()   # not required for basic vector copying

    for profile in DEMO_PROFILES:
        username = profile["username"]
        field = profile["field"]
        subjects = profile["subjects"]

        # 1) Guarantee a UUID is present in users1.db for this username
        # get_or_create_uuid should return the UUID string for this username
        try:
            uuid = get_or_create_uuid(username)
            logger.info("Mapped username '%s' -> uuid '%s'", username, uuid)
        except Exception as e:
            logger.error("Failed to get/create uuid for %s: %s", username, e)
            continue

        # 2) Ensure a UserProfile exists (create if not present)
        user_profile = user_mgr.get_user(uuid)
        if user_profile is None:
            try:
                # create_user expects a user_id string - pass the uuid
                user_profile = user_mgr.create_user(uuid)
                logger.info("Created UserProfile for uuid %s", uuid)
            except Exception as e:
                logger.error("Failed to create user profile for %s: %s", uuid, e)
                continue
        else:
            logger.info("Found existing UserProfile for uuid %s", uuid)

        # 3) Gather curated/arxiv IDs for the user's field and fetch the real vectors from Qdrant
        curated = CURATED_PAPERS_BY_FIELD.get(field, [])
        liked_papers_with_vectors = []
        found_count = 0
        for paper in curated:
            arxiv_id = paper["arxiv_id"]
            vec = fetch_paper_vector_from_qdrant(q_client, TITLE_COLLECTION_NAME, arxiv_id)
            if vec is not None:
                liked_papers_with_vectors.append((arxiv_id, vec))
                found_count += 1
            else:
                logger.warning("Paper %s (field=%s) not found in title collection '%s'", arxiv_id, field, TITLE_COLLECTION_NAME)

        logger.info("For user %s found %d/%d curated paper vectors in Qdrant", username, found_count, len(curated))

        # 4) Onboard the user locally with the 3 subject names and the liked papers
        # We'll pick up to 3 subject names (pad/truncate)
        s1 = subjects[0] if len(subjects) > 0 else "General"
        s2 = subjects[1] if len(subjects) > 1 else "General"
        s3 = subjects[2] if len(subjects) > 2 else "General"

        try:
            # user_profile.onboard_user(subject1_name, subject2_name, subject3_name, liked_papers_with_vectors, subject_keywords)
            user_profile.onboard_user(s1, s2, s3, liked_papers_with_vectors, None)
            logger.info("Onboarded user %s (uuid=%s) with subjects [%s, %s, %s]", username, uuid, s1, s2, s3)
        except Exception as e:
            logger.error("Failed to onboard user %s: %s", uuid, e)
            # continue, still attempt to upsert vector

        # 5) Build an initial user embedding vector
        if liked_papers_with_vectors:
            vectors = np.stack([vec for _, vec in liked_papers_with_vectors], axis=0)
            user_vec = np.mean(vectors, axis=0)
            # Normalize
            norm = np.linalg.norm(user_vec)
            if norm > 0:
                user_vec = user_vec / norm
        else:
            # fallback - generate a small random vector or zero vector; zero vector is safe but often not recommended
            user_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            logger.warning("No paper vectors found for user %s; using zero vector fallback.", uuid)

        # 6) Upsert the user vector into Qdrant user collection
        try:
            upsert_user_vector_to_qdrant(q_client, USER_COLLECTION_NAME, uuid, user_vec)
        except Exception as e:
            logger.error("Failed to upsert vector for user %s: %s", uuid, e)
            continue

        logger.info("✅ Finished initialization for user %s (uuid=%s). liked_papers=%d", username, uuid, len(liked_papers_with_vectors))

    logger.info("All demo users processed. Exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Exiting.")
        sys.exit(0)
