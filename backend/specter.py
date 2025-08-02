#!/usr/bin/env python3
# specter2_search_enhanced.py - FIXED VERSION

import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from qdrant_client import QdrantClient, models
import time
import warnings
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
from functools import lru_cache
import re
import logging
from sentence_transformers import SentenceTransformer
import csv 
import os


warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Your Qdrant Configuration
QDRANT_URL = "https://6b25695f-de3c-4dbd-bb36-6de748ff47f2.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ug0KQAaAKM7Hv-L3NprJnvuLgNcNL9D9847dfWRL_Fk"
COLLECTION_NAME = "arxiv_specter2_recommendations"

# ───────────  MiniLM TITLE-ONLY collection  ───────────
QDRANT_TITLE_URL      = "https://ba0f9774-1b9e-4b0b-bb05-db8fadfe122c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_TITLE_API_KEY  = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sMVFQwd_dg3z89uIih5r5olFlbXLAjl_Gcx0V5IJG-U"
TITLE_COLLECTION_NAME = "arxiv_papers_titles"
MINILM_MODEL_NAME     = "sentence-transformers/all-MiniLM-L6-v2"
TITLE_EMB_DIM         = 384
# ArXiv API Configuration
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
REQUEST_DELAY = 3.0

@dataclass
class PaperMetadata:
    """Structure to hold paper metadata"""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    categories: List[str]
    doi: Optional[str] = None
    journal_ref: Optional[str] = None

class ArXivMetadataFetcher:
    """Enhanced ArXiv metadata fetcher with better ID cleaning"""
    
    def __init__(self):
        self.last_request_time = 0
        self.cache = {}
        self.lock = threading.Lock()
    
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Enhanced ArXiv ID cleaning with better validation"""
        if not arxiv_id:
            return arxiv_id
            
        # Remove prefix which is arxiv: ...
        arxiv_id = re.sub(r'^(arxiv:|arXiv:)', '', arxiv_id, flags=re.IGNORECASE)
        
        # there might be versions in arxivid which isnt imp 
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        
        # if space is there
        arxiv_id = arxiv_id.strip()
        
        # if bad arxiv id's since qdrant reduces the data with leading 0's and trailing 0's
        if re.match(r'^\d{4}\.\d{4,5}$', arxiv_id):
            # Already in correct format then no problem .. but i am missing the one with 4 inputs but should actually be 5 ###### imp problem 
            return arxiv_id
        elif re.match(r'^\d{4}\.\d{1,3}$', arxiv_id):
            parts = arxiv_id.split('.')
            if len(parts[1]) < 4:
                # Pad with leading 0's since now it become xxxx.11 to xxxx.1100
                
                arxiv_id = f"{parts[0]}.{parts[1] + {'0'*(4 - len(parts[1]))}}"
                logger.info(f"🔧 Fixed ArXiv ID: {arxiv_id}")
        elif re.match(r'^\d{3}\.\d{4}$', arxiv_id):
            # adds trailing 0 to the LHS side 
            arxiv_id = f"0{arxiv_id}"
            logger.info(f"🔧 Fixed ArXiv ID: {arxiv_id}")
        elif re.match(r'^[a-z-]+/\d{7}$', arxiv_id):
            # Subject class format (e.g., math/9502222, cs/9301114)
            # old format thing 
            pass
        else:
            logger.warning(f"⚠️ Potentially invalid ArXiv ID format: {arxiv_id}")
        
        return arxiv_id
    
    def _validate_arxiv_ids(self, arxiv_ids: List[str]) -> List[str]:
        """Validate and filter out invalid ArXiv IDs"""
        valid_ids = []
        
        for arxiv_id in arxiv_ids:
            cleaned = self._clean_arxiv_id(arxiv_id)
            
            # Check if ID looks valid
            if (re.match(r'^\d{4}\.\d{4,5}$', cleaned) or # New format (YYMM.NNNN or YYMM.NNNNN) 
                re.match(r'^[a-z-]+/\d{7}$', cleaned) or  # Old format with subject
                re.match(r'^\d{7}$', cleaned)):  # Old format numeric only
                valid_ids.append(cleaned)
            else:
                logger.warning(f"⚠️ Skipping invalid ArXiv ID: {arxiv_id} -> {cleaned}")
        
        return valid_ids
    def _rate_limit(self):
        """Ensure we don't exceed ArXiv API rate limits"""
        with self.lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < REQUEST_DELAY:
                sleep_time = REQUEST_DELAY - elapsed
                logger.info(f"⏳ Rate limiting: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    
    def _parse_entry(self, entry, arxiv_id: str) -> PaperMetadata:
        """Parse ArXiv API XML entry into PaperMetadata object"""
        try:
            # Extract title
            title_elem = entry.find('{http://www.w3.org/2005/Atom}title')
            title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
            
            # Extract authors
            authors = []
            for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
                name_elem = author.find('{http://www.w3.org/2005/Atom}name')
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract abstract
            summary_elem = entry.find('{http://www.w3.org/2005/Atom}summary')
            abstract = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Extract published date
            published_elem = entry.find('{http://www.w3.org/2005/Atom}published')
            published = published_elem.text.strip() if published_elem is not None else ""
            
            # Extract categories
            categories = []
            for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            
            # Extract DOI (optional)
            doi = None
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.get('title') == 'doi':
                    doi = link.get('href')
                    break
            
            # Extract journal reference (optional)
            journal_ref = None
            journal_elem = entry.find('{http://arxiv.org/schemas/atom}journal_ref')
            if journal_elem is not None:
                journal_ref = journal_elem.text.strip()
            
            return PaperMetadata(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                categories=categories,
                doi=doi,
                journal_ref=journal_ref
            )
            
        except Exception as e:
            logger.error(f"❌ Error parsing entry for {arxiv_id}: {e}")
            # Return minimal metadata on error
            return PaperMetadata(
                arxiv_id=arxiv_id,
                title="Parse Error",
                authors=[],
                abstract="",
                published="",
                categories=[]
            )
            
            
    def fetch_batch_metadata(self, arxiv_ids: List[str]) -> Dict[str, PaperMetadata]:
        """Enhanced batch metadata fetching with better error handling"""
        results = {}
        
        # FIX 2: Validate IDs before making requests
        valid_ids = self._validate_arxiv_ids(arxiv_ids)
        uncached_ids = [aid for aid in valid_ids if aid not in self.cache]
        
        logger.info(f"📚 Fetching metadata for {len(uncached_ids)} papers (cached: {len(valid_ids) - len(uncached_ids)}, invalid: {len(arxiv_ids) - len(valid_ids)})...")
        
        if not uncached_ids:
            # Return cached results
            for arxiv_id in valid_ids:
                if arxiv_id in self.cache:
                    results[arxiv_id] = self.cache[arxiv_id]
            return results
        
        # FIX 3: Smaller batch sizes for better reliability
        batch_size = 5  # Reduced from 10
        
        for i in range(0, len(uncached_ids), batch_size):
            batch = uncached_ids[i:i + batch_size]
            
            try:
                self._rate_limit()
                id_list = ','.join(batch)
                url = f"{ARXIV_API_BASE}?id_list={id_list}"
                
                logger.info(f"🔍 Fetching batch {i//batch_size + 1}: {id_list}")
                
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                root = ET.fromstring(response.content)
                
                # Check for errors in response
                error_elem = root.find('.//{http://www.w3.org/2005/Atom}error')
                if error_elem is not None:
                    logger.error(f"❌ ArXiv API error: {error_elem.text}")
                    continue
                
                found_ids = set()
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    if id_elem is not None:
                        full_id = id_elem.text
                        arxiv_id = full_id.split('/')[-1]
                        arxiv_id = self._clean_arxiv_id(arxiv_id)
                        
                        metadata = self._parse_entry(entry, arxiv_id)
                        self.cache[arxiv_id] = metadata
                        results[arxiv_id] = metadata
                        found_ids.add(arxiv_id)
                
                # Log missing IDs
                missing_ids = set(batch) - found_ids
                if missing_ids:
                    logger.warning(f"⚠️ Missing IDs in batch: {missing_ids}")
                
                logger.info(f"✅ Batch {i//batch_size + 1}: {len(found_ids)}/{len(batch)} papers found")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Network error in batch {i//batch_size + 1}: {e}")
                continue
            except ET.ParseError as e:
                logger.error(f"❌ XML parsing error in batch {i//batch_size + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"❌ Unexpected error in batch {i//batch_size + 1}: {e}")
                continue
        
        # Add cached results
        for arxiv_id in valid_ids:
            if arxiv_id in self.cache:
                results[arxiv_id] = self.cache[arxiv_id]
        
        return results
    
    
class SPECTER2Search:
    def __init__(self):
        """Initialize SPECTER2 search system with metadata fetching"""
        self.model = None
        self.tokenizer = None
        self.client = None
        self.metadata_fetcher = ArXivMetadataFetcher()
        self.title_model: Optional[SentenceTransformer] = None
        self.title_client: Optional[QdrantClient] = None
        self.collection_name = COLLECTION_NAME
        
        # Initialize components
        self._setup_model()             # specter2 download
        self._setup_client()            # specter2 qdrant client
        self._setup_title_components()  # minilm download and title only qdrant
        
    # specter2 download
    def _setup_model(self):
        """Setup SPECTER2 model optimized for inference"""
        logger.info("🔬 Setting up SPECTER2 model for search...")
        try:
            logger.info("📥 Loading SPECTER2 base model...")
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
            self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            
            logger.info("🔧 Loading SPECTER2 proximity adapter...")
            self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
            
            logger.info("✅ SPECTER2 ready with CPU")
            
        except Exception as e:
            logger.error(f"❌ Error loading SPECTER2: {e}")
            raise
    #specter2 on 
    def _setup_client(self):
        """Setup main Qdrant client"""
        try:
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=120
            )
            logger.info("✅ Connected to main Qdrant vector database")
            
            try:
                collection_info = self.client.get_collection(COLLECTION_NAME)
                logger.info(f"✅ Collection '{COLLECTION_NAME}' found with {collection_info.points_count} points")
            except Exception as e:
                logger.warning(f"⚠️ Could not get collection info: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to main Qdrant: {e}")
            raise
    
    def _setup_title_components(self):
        """Setup title model and client with proper error handling"""
        logger.info("🔧 Setting up title search components...")
        
        # Setup title model
        try:
            logger.info("🔬 Loading MiniLM title encoder...")
            self.title_model = SentenceTransformer(MINILM_MODEL_NAME)
            
            
            logger.info("✅ Title model using CPU")
                
            logger.info("✅ MiniLM title model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load title model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.title_model = None
        
        # Setup title client
        try:
            logger.info("🔗 Setting up title client...")
            self.title_client = QdrantClient(
                url=QDRANT_TITLE_URL,
                api_key=QDRANT_TITLE_API_KEY,
                timeout=120
            )
            
            # Test connection
            collection_info = self.title_client.get_collection(TITLE_COLLECTION_NAME)
            logger.info(f"✅ Title collection '{TITLE_COLLECTION_NAME}' found with {collection_info.points_count} points")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup title client: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.title_client = None
            
        # Summary
        if self.title_model is not None and self.title_client is not None:
            logger.info("🎉 Title search components ready!")
        else:
            logger.warning("⚠️ Title search components not available, will use SPECTER2 fallback")
    
    def _generate_title_embedding(self, text: str) -> np.ndarray:
        """Generate title embedding using MiniLM"""
        if self.title_model is None:
            raise RuntimeError("Title model not initialized")
        
        try:
            vec = self.title_model.encode(
                text, 
                convert_to_numpy=True,
                device="cpu"
            )
            return vec / np.linalg.norm(vec)
        except Exception as e:
            logger.error(f"❌ Error generating title embedding: {e}")
            raise
    
    # Add 'return_vector=False' to the signature
    def search_titles(self, query_text: str, top_k: int = 10, fetch_metadata: bool = True, return_vector: bool = False):
        """Search using title-only MiniLM embeddings"""
        logger.info(f"📚 TITLE SEARCH for: '{query_text}'")
        
        if self.title_model is None:
            raise RuntimeError("Title model not available")
        
        if self.title_client is None:
            raise RuntimeError("Title client not available")
        
        try:
            logger.info("🔢 Generating title embedding...")
            qvec = self._generate_title_embedding(query_text)
            logger.info(f"✅ Embedding generated, shape: {qvec.shape}")
            
            logger.info("🔍 Executing title search...")
            t0 = time.time()
            
            results = self.title_client.search(
                collection_name=TITLE_COLLECTION_NAME,
                query_vector=qvec.astype(np.float32).tolist(),
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=64),
                with_payload=True,
                with_vectors=return_vector # <-- ADD THIS LINE
            )
            
            search_time = (time.time() - t0) * 1000
            logger.info(f"✅ Title search completed in {search_time:.1f}ms")
            logger.info(f"📊 Found {len(results)} results")
            
            metadata = self._maybe_fetch_metadata(results, fetch_metadata)
            
            return results, search_time, metadata
            
        except Exception as e:
            logger.error(f"❌ Error in title search: {e}")
            raise
        
    
    def find_similar_by_id(self, arxiv_id: str, top_k: int = 10, fetch_metadata: bool = True, return_vector: bool = False):
        """
        Finds papers similar to a given paper using its arXiv ID.
        
        1. Fetches the vector for the given arxiv_id.
        2. Uses that vector to perform a search for similar vectors.
        3. Filters out the original paper from the results.
        """
        logger.info(f"🔎 Finding papers similar to ArXiv ID: {arxiv_id}...")
        
        # Step 1: Retrieve the vector for the source paper
        try:
            # We use a filter to find the paper by its arxiv_id in the payload
            source_papers = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="arxiv_id", match=models.MatchValue(value=arxiv_id))]
                ),
                limit=1,
                with_vectors=True
            )
            
            if not source_papers[0]:
                logger.error(f"❌ Could not find source paper with ArXiv ID: {arxiv_id}")
                return [], 0.0, {}

            source_vector = source_papers[0][0].vector
            
        # This is the corrected version
        except Exception as e:
            logger.error(f"❌ Error fetching source vector for {arxiv_id}: {e}")
            # Raise a standard Python error. The API layer (app.py) will handle it.
            raise ValueError(f"Paper with ArXiv ID '{arxiv_id}' not found in the database.")
        # Step 2: Use the fetched vector to find similar papers.
        # We fetch k+1 results to have a buffer in case the source paper is returned.
        start_time = time.time()
        similar_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=source_vector,
            limit=top_k + 1,
            with_payload=True,
            with_vectors=return_vector
        )
        search_time = (time.time() - start_time) * 1000

        # Step 3: Filter out the original paper from the results list
        final_results = [
            result for result in similar_results 
            if result.payload.get('arxiv_id') != arxiv_id
        ][:top_k] # Ensure we only return top_k results

        logger.info(f"✅ Found {len(final_results)} similar papers in {search_time:.0f}ms.")

        # Step 4: Fetch metadata for the similar papers
        metadata_dict = self._maybe_fetch_metadata(final_results, fetch_metadata)

        return final_results, search_time, metadata_dict
    
    def _generate_query_embedding(self, query_text):
        """Generate SPECTER2 embedding for query"""
        inputs = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            return_token_type_ids=False
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        return query_embedding
    
    # Add 'return_vector=False' to the signature
    def search(self, query_text, top_k=10, search_mode="balanced", fetch_metadata=True, return_vector: bool = False):
        """Search papers using SPECTER2 embeddings"""
        ef_configs = {
            'fast': 32,
            'balanced': 64,
            'quality': 128
        }
        ef_search = ef_configs.get(search_mode, 64)
        
        query_embedding = self._generate_query_embedding(query_text)
        
        start_time = time.time()
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.astype(np.float32).tolist(),
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=ef_search),
                with_payload=True,
                with_vectors=return_vector 
            )
            search_time = (time.time() - start_time) * 1000
            logger.info(f"✅ SPECTER2 search completed in {search_time:.0f}ms, found {len(results)} results")
            
        except Exception as e:
            logger.error(f"❌ SPECTER2 search failed: {e}")
            raise
        
        metadata_dict = {}
        if fetch_metadata and results:
            arxiv_ids = [r.payload.get('arxiv_id', '') for r in results if r.payload.get('arxiv_id')]
            if arxiv_ids:
                metadata_dict = self.metadata_fetcher.fetch_batch_metadata(arxiv_ids)
        
        return results, search_time, metadata_dict
    
    
    
    def auto_search(self, query_text: str, top_k: int = 10,
                   fetch_metadata: bool = True,
                   title_score_th: float = 0.7 , 
                   return_vector: bool = False,
                   show_vectors: bool = False):  # FIX 4: Lower threshold
        """Auto search with more reasonable threshold"""
        
        def looks_like_title(q: str) -> bool:
            """Enhanced title detection"""
            tokens = q.strip().split()
            word_count = len(tokens)
            char_count = len(q)
            starts_uppercase = q[:1].isupper()
            
            # FIX 5: More flexible title detection
            has_common_words = any(word.lower() in ['paper', 'study', 'analysis', 'review'] 
                                 for word in tokens)
            
            # Accept as title if:
            # - Short and starts with uppercase, OR
            # - Contains common academic words
            result = ((word_count <= 8 or char_count <= 60) and starts_uppercase) or has_common_words
            
            logger.info(f"🔍 ENHANCED TITLE DETECTION for: '{q}'")
            logger.info(f"   📊 Word count: {word_count} (≤8? {word_count <= 8})")
            logger.info(f"   📊 Character count: {char_count} (≤60? {char_count <= 60})")
            logger.info(f"   📊 Starts uppercase: {starts_uppercase}")
            logger.info(f"   📊 Has common words: {has_common_words}")
            logger.info(f"   ✅ Result: {result}")
            
            return result
        
        logger.info(f"🚀 ENHANCED AUTO_SEARCH started for: '{query_text}'")
        logger.info(f"   📋 Parameters: top_k={top_k}, fetch_metadata={fetch_metadata}, title_score_th={title_score_th}")
        
        title_detected = looks_like_title(query_text)
        
        # Try title search with lower threshold
        if title_detected and self.title_model is not None and self.title_client is not None:
            try:
                logger.info("🔎 AUTO: Trying title-only MiniLM search...")
                
                t_res, t_ms, meta = self.search_titles(query_text, top_k, fetch_metadata, return_vector=return_vector)
                
                if t_res:
                    top_score = t_res[0].score
                    logger.info(f"   🎯 Top result score: {top_score:.6f}")
                    logger.info(f"   ✅ Score threshold check: {top_score} >= {title_score_th} = {top_score >= title_score_th}")
                    
                    if top_score >= title_score_th:
                        logger.info(f"✅ TITLE path accepted (score {top_score:.3f})")
                        return t_res, t_ms, "title", meta
                    else:
                        logger.info(f"↩️ Fallback to SPECTER (low score: {top_score:.3f} < {title_score_th})")
                        
            except Exception as e:
                logger.error(f"❌ Title search failed: {e}")
        
        # Fallback to SPECTER2 with faster mode
        logger.info("🔄 Using SPECTER2 search...")
        s_res, s_ms, meta = self.search(query_text, top_k, "fast", fetch_metadata, return_vector=return_vector)  # FIX 6: Use fast mode
        return s_res, s_ms, "specter", meta

    def _maybe_fetch_metadata(self, results, fetch_metadata):
        """Helper to fetch metadata if requested"""
        if not fetch_metadata or not results:
            return {}
        
        arxiv_ids = [r.payload.get('arxiv_id', '') for r in results if r.payload.get('arxiv_id')]
        if arxiv_ids:
            return self.metadata_fetcher.fetch_batch_metadata(arxiv_ids)
        return {}
    
    def smart_search(self, query_text, top_k=10, min_good_results=3, fetch_metadata=True, return_vector: bool = False):
        """Intelligent search that automatically adjusts efSearch for optimal results"""
        logger.info(f"🧠 Smart search for: '{query_text}'")
        
        results, search_time, metadata = self.search(query_text, top_k, "fast", fetch_metadata, return_vector=return_vector)
        good_results = len([r for r in results if r.score > 0.7])
        
        if good_results >= min_good_results:
            logger.info(f"✅ Fast search sufficient: {good_results} good results in {search_time:.0f}ms")
            return results, search_time, "fast", metadata
        
        logger.info(f"🔄 Escalating to balanced search...")
        results, search_time, metadata = self.search(query_text, top_k, "balanced", fetch_metadata, return_vector=return_vector)
        good_results = len([r for r in results if r.score > 0.7])
        
        if good_results >= min_good_results:
            logger.info(f"✅ Balanced search sufficient: {good_results} good results in {search_time:.0f}ms")
            return results, search_time, "balanced", metadata
        
        logger.info(f"🔄 Escalating to quality search...")
        results, search_time, metadata = self.search(query_text, top_k, "quality", fetch_metadata, return_vector=return_vector)
        good_results = len([r for r in results if r.score > 0.7])
        logger.info(f"✅ Quality search completed: {good_results} good results in {search_time:.0f}ms")
        
        return results, search_time, "quality", metadata
    
    def compare_search_modes(self, query_text, top_k=10, fetch_metadata=True, return_vector: bool = False):
        """Compare all search modes"""
        logger.info(f"🔍 COMPARING SEARCH MODES for: '{query_text}'")
        
        mode_results = {}
        all_metadata = {}
        
        for mode in ["fast", "balanced", "quality"]:
            logger.info(f"--- {mode.upper()} MODE ---")
            results, search_time, metadata = self.search(query_text, top_k, mode, fetch_metadata, return_vector=return_vector)
            
            all_metadata.update(metadata)
            
            high_confidence = len([r for r in results if r.score > 0.8])
            good_results = len([r for r in results if r.score > 0.7])
            avg_score = np.mean([r.score for r in results]) if results else 0
            
            mode_results[mode] = {
                'results': results,
                'search_time': search_time,
                'high_confidence': high_confidence,
                'good_results': good_results,
                'avg_score': avg_score,
                'metadata': metadata
            }
            
            logger.info(f"⚡ Time: {search_time:.0f}ms")
            logger.info(f"🎯 High confidence (>0.8): {high_confidence}")
            logger.info(f"✅ Good results (>0.7): {good_results}")
            logger.info(f"📊 Average score: {avg_score:.3f}")
        
        best_mode = self._recommend_best_mode(mode_results)
        logger.info(f"🏆 Best mode for this query: **{best_mode.upper()}**")
        
        return mode_results, best_mode, all_metadata
    
    def _recommend_best_mode(self, mode_results):
        """Recommend the best search mode based on results"""
        if mode_results['fast']['good_results'] >= 3:
            return 'fast'
        
        if mode_results['balanced']['good_results'] > mode_results['fast']['good_results'] * 1.2:
            return 'balanced'
        
        return 'quality'
    
    def print_results(self, results, search_time, query_text, metadata_dict=None, mode="", show_vectors=False):
        """Pretty print search results with metadata and optionally vectors"""
        logger.info(f"🔍 SPECTER2 Results for: '{query_text}'")
        logger.info(f"⚡ Search completed in {search_time:.0f}ms {mode}")
        logger.info(f"📊 Found {len(results)} papers")
        
        for i, result in enumerate(results, 1):
            confidence = "🔥" if result.score > 0.8 else "✅" if result.score > 0.7 else "⚠️"
            arxiv_id = result.payload.get('arxiv_id', 'Unknown')
            
            # Get metadata if available
            paper_metadata = None
            if metadata_dict:
                clean_id = self.metadata_fetcher._clean_arxiv_id(arxiv_id)
                paper_metadata = metadata_dict.get(clean_id)
            
            logger.info(f"{i:2d}. {confidence} Score: {result.score:.4f}")
            logger.info(f"    📄 ArXiv ID: {arxiv_id}")
            
            # Show vector if requested and available
            if show_vectors and hasattr(result, 'vector') and result.vector is not None:
                vector_preview = result.vector[:10] if len(result.vector) > 10 else result.vector
                logger.info(f"    🔢 Vector (dim={len(result.vector)}): {vector_preview}...")
                # Optionally show vector stats
                vector_array = np.array(result.vector)
                logger.info(f"    📊 Vector stats: mean={vector_array.mean():.6f}, std={vector_array.std():.6f}, norm={np.linalg.norm(vector_array):.6f}")
            
            if paper_metadata:
                logger.info(f"    📚 Title: {paper_metadata.title}")
                
                # Authors (limit to first 3)
                if paper_metadata.authors:
                    authors_str = ", ".join(paper_metadata.authors[:3])
                    if len(paper_metadata.authors) > 3:
                        authors_str += f" et al. ({len(paper_metadata.authors)} authors)"
                    logger.info(f"    👥 Authors: {authors_str}")
                
                # Published date
                if paper_metadata.published:
                    pub_date = paper_metadata.published.split('T')[0]  # Extract date part
                    logger.info(f"    📅 Published: {pub_date}")
                
                # Categories
                if paper_metadata.categories:
                    cats = ", ".join(paper_metadata.categories[:3])
                    logger.info(f"    🏷️  Categories: {cats}")
                
                # DOI if available
                if paper_metadata.doi:
                    logger.info(f"    🔗 DOI: {paper_metadata.doi}")
                
                # Journal reference if available
                if paper_metadata.journal_ref:
                    logger.info(f"    📖 Journal: {paper_metadata.journal_ref}")
                
                # Abstract preview (first 200 chars)
                if paper_metadata.abstract:
                    abstract_preview = paper_metadata.abstract[:200]
                    if len(paper_metadata.abstract) > 200:
                        abstract_preview += "..."
                    logger.info(f"    📝 Abstract: {abstract_preview}")
            else:
                logger.info(f"    ⚠️  Metadata not available")
            
            logger.info("")  # Empty line for readability
    def analyze_vectors(self, results, query_vector=None):
        """Analyze and compare vectors from search results"""
        if not results or not hasattr(results[0], 'vector') or results[0].vector is None:
            logger.warning("⚠️ No vectors available in results")
            return
        
        logger.info("📊 VECTOR ANALYSIS")
        logger.info("=" * 50)
        
        # Extract vectors
        vectors = [np.array(r.vector) for r in results if hasattr(r, 'vector') and r.vector is not None]
        
        if not vectors:
            logger.warning("⚠️ No valid vectors found")
            return
        
        # Basic stats
        vector_dim = len(vectors[0])
        logger.info(f"🔢 Vector dimension: {vector_dim}")
        logger.info(f"📊 Number of vectors: {len(vectors)}")
        
        # Vector norms
        norms = [np.linalg.norm(v) for v in vectors]
        logger.info(f"📏 Vector norms - Mean: {np.mean(norms):.6f}, Std: {np.std(norms):.6f}")
        
        # If query vector provided, compute similarities
        if query_vector is not None:
            query_vec = np.array(query_vector)
            similarities = [np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v)) 
                        for v in vectors]
            logger.info(f"🎯 Cosine similarities with query:")
            for i, sim in enumerate(similarities[:5]):  # Show top 5
                arxiv_id = results[i].payload.get('arxiv_id', 'Unknown')
                logger.info(f"    {i+1}. {arxiv_id}: {sim:.6f}")
        
        # Compute pairwise similarities between top results
        if len(vectors) >= 2:
            logger.info(f"🔗 Pairwise similarities (top 3 results):")
            for i in range(min(3, len(vectors))):
                for j in range(i+1, min(3, len(vectors))):
                    sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]))
                    id_i = results[i].payload.get('arxiv_id', f'Result_{i+1}')
                    id_j = results[j].payload.get('arxiv_id', f'Result_{j+1}')
                    logger.info(f"    {id_i} ↔ {id_j}: {sim:.6f}")

    def save_vectors_to_file(self, results, filename="search_vectors.npz", include_metadata=True):
        """Save vectors and metadata to file for analysis"""
        if not results or not hasattr(results[0], 'vector') or results[0].vector is None:
            logger.warning("⚠️ No vectors to save")
            return
        
        # Extract vectors and metadata
        vectors = []
        metadata = []
        
        for result in results:
            if hasattr(result, 'vector') and result.vector is not None:
                vectors.append(result.vector)
                if include_metadata:
                    meta = {
                        'arxiv_id': result.payload.get('arxiv_id', ''),
                        'score': result.score,
                        'payload': dict(result.payload)
                    }
                    metadata.append(meta)
        
        # Save to file
        vectors_array = np.array(vectors)
        save_dict = {'vectors': vectors_array}
        
        if include_metadata:
            save_dict['metadata'] = metadata
        
        np.savez_compressed(filename, **save_dict)
        logger.info(f"💾 Saved {len(vectors)} vectors to {filename}")
        logger.info(f"📊 Vector shape: {vectors_array.shape}")

    def compare_vector_distributions(self, results1, results2, label1="Search 1", label2="Search 2"):
        """Compare vector distributions between two search results"""
        def get_vector_stats(results, label):
            vectors = [np.array(r.vector) for r in results 
                    if hasattr(r, 'vector') and r.vector is not None]
            if not vectors:
                return None
            
            vectors_array = np.array(vectors)
            stats = {
                'mean_norm': np.mean([np.linalg.norm(v) for v in vectors]),
                'std_norm': np.std([np.linalg.norm(v) for v in vectors]),
                'mean_values': np.mean(vectors_array, axis=0),
                'std_values': np.std(vectors_array, axis=0),
                'count': len(vectors)
            }
            
            logger.info(f"📊 {label} Vector Statistics:")
            logger.info(f"    Count: {stats['count']}")
            logger.info(f"    Mean norm: {stats['mean_norm']:.6f}")
            logger.info(f"    Std norm: {stats['std_norm']:.6f}")
            logger.info(f"    Mean value: {np.mean(stats['mean_values']):.6f}")
            logger.info(f"    Std value: {np.mean(stats['std_values']):.6f}")
            
            return stats
        
        logger.info("🔍 COMPARING VECTOR DISTRIBUTIONS")
        logger.info("=" * 50)
        
        stats1 = get_vector_stats(results1, label1)
        stats2 = get_vector_stats(results2, label2)
        
        if stats1 and stats2:
            # Compare distributions
            norm_diff = abs(stats1['mean_norm'] - stats2['mean_norm'])
            logger.info(f"📈 Norm difference: {norm_diff:.6f}")
            
            # Compare mean vectors
            mean_similarity = np.dot(stats1['mean_values'], stats2['mean_values']) / \
                            (np.linalg.norm(stats1['mean_values']) * np.linalg.norm(stats2['mean_values']))
            logger.info(f"🎯 Mean vector similarity: {mean_similarity:.6f}")

    
    def save_embeddings_to_csv(self, results, query_text, directory="search_embeddings"):
        """Saves the ArXiv ID and embedding vector for each result to a CSV file."""
        if not results or not hasattr(results[0], 'vector') or not results[0].vector:
            logger.warning("No results with vectors to save to CSV.")
            return None

        # --- Create a safe filename ---
        safe_query = "".join(c for c in query_text if c.isalnum() or c in (' ', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')
        filename = f"embeddings_{safe_query[:50]}.csv"

        # --- Ensure the directory exists ---
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"📁 Created directory: {directory}")
        
        filepath = os.path.join(directory, filename)

        # --- Define CSV headers ---
        # The first header is 'arxiv_id', the rest are for the vector dimensions
        vector_dim = len(results[0].vector)
        headers = ['arxiv_id'] + [f'dim_{i}' for i in range(vector_dim)]

        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers) # Write the header row

                for result in results:
                    # Ensure the result has a vector before trying to save
                    if hasattr(result, 'vector') and result.vector:
                        arxiv_id = result.payload.get('arxiv_id', 'N/A')
                        # Create a row with the ID followed by all vector elements
                        row = [arxiv_id] + result.vector
                        writer.writerow(row)
            
            logger.info(f"💾 Successfully saved embeddings for {len(results)} results to {filepath}")
            return filepath
        except IOError as e:
            logger.error(f"❌ Failed to write CSV file at {filepath}: {e}")
            return None