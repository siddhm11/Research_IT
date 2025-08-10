#!/usr/bin/env python3
# specter2_search_clean.py - Streamlined version with core functionality only

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
                
                arxiv_id = f"{parts[0]}.{parts[1] + '0'*(4 - len(parts[1]))}"
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
        
        # Validate IDs before making requests
        valid_ids = self._validate_arxiv_ids(arxiv_ids)
        uncached_ids = [aid for aid in valid_ids if aid not in self.cache]
        
        logger.info(f"📚 Fetching metadata for {len(uncached_ids)} papers (cached: {len(valid_ids) - len(uncached_ids)}, invalid: {len(arxiv_ids) - len(valid_ids)})...")
        
        if not uncached_ids:
            # Return cached results
            for arxiv_id in valid_ids:
                if arxiv_id in self.cache:
                    results[arxiv_id] = self.cache[arxiv_id]
            return results
        
        # Smaller batch sizes for better reliability
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
                with_vectors=return_vector
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
            
        except Exception as e:
            logger.error(f"❌ Error fetching source vector for {arxiv_id}: {e}")
            raise ValueError(f"Paper with ArXiv ID '{arxiv_id}' not found in the database.")
            
        # Step 2: Use the fetched vector to find similar papers
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
        ][:top_k]

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
                   return_vector: bool = False):
        """Auto search with title detection and fallback to SPECTER2"""
        
        def looks_like_title(q: str) -> bool:
            """Enhanced title detection"""
            tokens = q.strip().split()
            word_count = len(tokens)
            char_count = len(q)
            starts_uppercase = q[:1].isupper()
            
            # More flexible title detection
            has_common_words = any(word.lower() in ['paper', 'study', 'analysis', 'review'] 
                                 for word in tokens)
            
            # Accept as title if:
            # - Short and starts with uppercase, OR
            # - Contains common academic words
            result = ((word_count <= 8 or char_count <= 60) and starts_uppercase) or has_common_words
            
            logger.info(f"🔍 TITLE DETECTION for: '{q}'")
            logger.info(f"   📊 Word count: {word_count} (≤8? {word_count <= 8})")
            logger.info(f"   📊 Character count: {char_count} (≤60? {char_count <= 60})")
            logger.info(f"   📊 Starts uppercase: {starts_uppercase}")
            logger.info(f"   📊 Has common words: {has_common_words}")
            logger.info(f"   ✅ Result: {result}")
            
            return result
        
        logger.info(f"🚀 AUTO_SEARCH started for: '{query_text}'")
        logger.info(f"   📋 Parameters: top_k={top_k}, fetch_metadata={fetch_metadata}, title_score_th={title_score_th}")
        
        title_detected = looks_like_title(query_text)
        
        # Try title search with threshold check
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
        
        # Fallback to SPECTER2 with fast mode
        logger.info("🔄 Using SPECTER2 search...")
        s_res, s_ms, meta = self.search(query_text, top_k, "fast", fetch_metadata, return_vector=return_vector)
        return s_res, s_ms, "specter", meta

    def _maybe_fetch_metadata(self, results, fetch_metadata):
        """Helper to fetch metadata if requested"""
        if not fetch_metadata or not results:
            return {}
        
        arxiv_ids = [r.payload.get('arxiv_id', '') for r in results if r.payload.get('arxiv_id')]
        if arxiv_ids:
            return self.metadata_fetcher.fetch_batch_metadata(arxiv_ids)
        return {}
    
    def print_results(self, results, search_time, query_text, metadata_dict=None, mode=""):
        """Pretty print search results with metadata"""
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