#!/usr/bin/env python3
# specter2_search_enhanced.py

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

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Your Qdrant Configuration
QDRANT_URL = "https://6b25695f-de3c-4dbd-bb36-6de748ff47f2.us-east-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Ug0KQAaAKM7Hv-L3NprJnvuLgNcNL9D9847dfWRL_Fk"
COLLECTION_NAME = "arxiv_specter2_recommendations"

# ArXiv API Configuration
ARXIV_API_BASE = "http://export.arxiv.org/api/query"
REQUEST_DELAY = 3.0  # ArXiv requests delay (3 seconds between requests)

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
    """Handles fetching metadata from ArXiv API with caching and rate limiting"""
    
    def __init__(self):
        self.last_request_time = 0
        self.cache = {}
        self.lock = threading.Lock()
    
    def _rate_limit(self):
        """Implement rate limiting for ArXiv API"""
        with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < REQUEST_DELAY:
                sleep_time = REQUEST_DELAY - time_since_last
                logger.info(f"⏳ Rate limiting: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean and normalize ArXiv ID"""
        # Remove any prefixes like 'arxiv:' or 'arXiv:'
        arxiv_id = re.sub(r'^(arxiv:|arXiv:)', '', arxiv_id, flags=re.IGNORECASE)
        # Remove version number (e.g., v1, v2)
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
        return arxiv_id.strip()
    
    def fetch_paper_metadata(self, arxiv_id: str) -> Optional[PaperMetadata]:
        """Fetch metadata for a single paper from ArXiv API"""
        cleaned_id = self._clean_arxiv_id(arxiv_id)
        
        # Check cache first
        if cleaned_id in self.cache:
            return self.cache[cleaned_id]
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Make API request
            url = f"{ARXIV_API_BASE}?id_list={cleaned_id}"
            logger.info(f"📡 Fetching metadata for {cleaned_id}...")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Find the entry
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            if entry is None:
                logger.warning(f"⚠️  No metadata found for {cleaned_id}")
                return None
            
            # Extract metadata
            metadata = self._parse_entry(entry, cleaned_id)
            
            # Cache the result
            self.cache[cleaned_id] = metadata
            
            return metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error fetching {cleaned_id}: {e}")
            return None
        except ET.ParseError as e:
            logger.error(f"❌ XML parsing error for {cleaned_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error fetching {cleaned_id}: {e}")
            return None
    
    def _parse_entry(self, entry, arxiv_id: str) -> PaperMetadata:
        """Parse XML entry into PaperMetadata"""
        ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
        
        # Extract title
        title_elem = entry.find('atom:title', ns)
        title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
        
        # Extract authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None:
                authors.append(name_elem.text.strip())
        
        # Extract abstract
        summary_elem = entry.find('atom:summary', ns)
        abstract = summary_elem.text.strip() if summary_elem is not None else ""
        
        # Extract published date
        published_elem = entry.find('atom:published', ns)
        published = published_elem.text.strip() if published_elem is not None else ""
        
        # Extract categories
        categories = []
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                categories.append(term)
        
        # Extract DOI (if available)
        doi = None
        doi_elem = entry.find('arxiv:doi', ns)
        if doi_elem is not None:
            doi = doi_elem.text.strip()
        
        # Extract journal reference (if available)
        journal_ref = None
        journal_elem = entry.find('arxiv:journal_ref', ns)
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
    
    def fetch_batch_metadata(self, arxiv_ids: List[str]) -> Dict[str, PaperMetadata]:
        """Fetch metadata for multiple papers efficiently"""
        results = {}
        
        # Filter out already cached items
        uncached_ids = [aid for aid in arxiv_ids if self._clean_arxiv_id(aid) not in self.cache]
        
        logger.info(f"📚 Fetching metadata for {len(uncached_ids)} papers (cached: {len(arxiv_ids) - len(uncached_ids)})...")
        
        # Batch API requests (ArXiv supports multiple IDs)
        batch_size = 10  # ArXiv recommends small batches
        
        for i in range(0, len(uncached_ids), batch_size):
            batch = uncached_ids[i:i + batch_size]
            cleaned_batch = [self._clean_arxiv_id(aid) for aid in batch]
            
            try:
                # Rate limiting
                self._rate_limit()
                
                # Make batch API request
                id_list = ','.join(cleaned_batch)
                url = f"{ARXIV_API_BASE}?id_list={id_list}"
                
                response = requests.get(url, timeout=15)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.content)
                
                # Process each entry
                for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                    # Extract ArXiv ID from entry
                    id_elem = entry.find('{http://www.w3.org/2005/Atom}id')
                    if id_elem is not None:
                        full_id = id_elem.text
                        # Extract just the ID part
                        arxiv_id = full_id.split('/')[-1]
                        arxiv_id = self._clean_arxiv_id(arxiv_id)
                        
                        # Parse and cache metadata
                        metadata = self._parse_entry(entry, arxiv_id)
                        self.cache[arxiv_id] = metadata
                        results[arxiv_id] = metadata
                
                logger.info(f"✅ Batch {i//batch_size + 1}: {len(batch)} papers processed")
                
            except Exception as e:
                logger.error(f"❌ Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        # Add cached results
        for arxiv_id in arxiv_ids:
            cleaned_id = self._clean_arxiv_id(arxiv_id)
            if cleaned_id in self.cache:
                results[cleaned_id] = self.cache[cleaned_id]
        
        return results

class SPECTER2Search:
    def __init__(self):
        """Initialize SPECTER2 search system with metadata fetching"""
        self.model = None
        self.tokenizer = None
        self.gpu_count = 0
        self.client = None
        self.metadata_fetcher = ArXivMetadataFetcher()
        self._setup_model()
        self._setup_client()
    
    def _setup_model(self):
        """Setup SPECTER2 model optimized for inference"""
        logger.info("🔬 Setting up SPECTER2 model for search...")
        
        # GPU detection
        if torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"✓ CUDA available - GPUs: {self.gpu_count}")
            for i in range(self.gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        else:
            logger.info("⚠️  No CUDA available, using CPU")
            self.gpu_count = 0
        
        try:
            # Load SPECTER2 base model and adapter
            logger.info("📥 Loading SPECTER2 base model...")
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
            self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
            
            logger.info("🔧 Loading SPECTER2 proximity adapter...")
            self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
            
            # GPU optimization
            if self.gpu_count > 1:
                logger.info(f"🚀 Enabling DataParallel across {self.gpu_count} GPUs")
                self.model = torch.nn.DataParallel(self.model)
                self.model = self.model.to('cuda')
                self.model.half()  # Float16 for memory efficiency
                logger.info("✅ SPECTER2 ready with multi-GPU support")
            elif self.gpu_count == 1:
                self.model = self.model.to('cuda')
                self.model.half()
                logger.info("✅ SPECTER2 ready with single GPU")
            else:
                logger.info("✅ SPECTER2 ready with CPU")
            
        except Exception as e:
            logger.error(f"❌ Error loading SPECTER2: {e}")
            logger.info("💡 Make sure 'adapters' library is installed: pip install adapters")
            raise
    
    def _setup_client(self):
        """Setup Qdrant client"""
        try:
            self.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=120
            )
            logger.info("✅ Connected to Qdrant vector database")
            
            # Test connection by getting collection info
            try:
                collection_info = self.client.get_collection(COLLECTION_NAME)
                logger.info(f"✅ Collection '{COLLECTION_NAME}' found with {collection_info.points_count} points")
            except Exception as e:
                logger.warning(f"⚠️ Could not get collection info: {e}")
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            raise
    
    def _generate_query_embedding(self, query_text):
        """Generate SPECTER2 embedding for query"""
        # Tokenize with SPECTER2 settings
        inputs = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
            return_token_type_ids=False
        )
        
        # Generate embedding with GPU optimization
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            if self.gpu_count > 1:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            else:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)
                query_embedding = outputs.last_hidden_state[:, 0, :].numpy()[0]
        
        # Normalize embedding for consistent similarity scores
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        return query_embedding
    
    def search(self, query_text, top_k=10, search_mode="balanced", fetch_metadata=True):
        """
        Search papers using SPECTER2 embeddings with optional metadata fetching
        
        Args:
            query_text (str): Search query
            top_k (int): Number of results to return
            search_mode (str): 'fast', 'balanced', or 'quality'
            fetch_metadata (bool): Whether to fetch paper metadata from ArXiv
            
        Returns:
            tuple: (results, search_time_ms, metadata_dict)
        """
        # Configure efSearch based on mode
        ef_configs = {
            'fast': 32,        # ~20-30ms, ~87% recall
            'balanced': 64,    # ~30-45ms, ~91% recall  
            'quality': 128     # ~45-65ms, ~95% recall
        }
        ef_search = ef_configs.get(search_mode, 64)
        
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query_text)
        
        # Search in Qdrant with optimized parameters
        start_time = time.time()
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding.astype(np.float32).tolist(),
                limit=top_k,
                search_params=models.SearchParams(hnsw_ef=ef_search),
                with_payload=True
            )
            search_time = (time.time() - start_time) * 1000
            logger.info(f"✅ Search completed in {search_time:.0f}ms, found {len(results)} results")
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            raise
        
        # Fetch metadata if requested
        metadata_dict = {}
        if fetch_metadata and results:
            arxiv_ids = [r.payload.get('arxiv_id', '') for r in results if r.payload.get('arxiv_id')]
            if arxiv_ids:
                metadata_dict = self.metadata_fetcher.fetch_batch_metadata(arxiv_ids)
        
        return results, search_time, metadata_dict
    
    def smart_search(self, query_text, top_k=10, min_good_results=3, fetch_metadata=True):
        """
        Intelligent search that automatically adjusts efSearch for optimal results
        """
        logger.info(f"🧠 Smart search for: '{query_text}'")
        
        # Start with fast search
        results, search_time, metadata = self.search(query_text, top_k, "fast", fetch_metadata)
        good_results = len([r for r in results if r.score > 0.7])
        
        if good_results >= min_good_results:
            logger.info(f"✅ Fast search sufficient: {good_results} good results in {search_time:.0f}ms")
            return results, search_time, "fast", metadata
        
        # Try balanced
        logger.info(f"🔄 Escalating to balanced search...")
        results, search_time, metadata = self.search(query_text, top_k, "balanced", fetch_metadata)
        good_results = len([r for r in results if r.score > 0.7])
        
        if good_results >= min_good_results:
            logger.info(f"✅ Balanced search sufficient: {good_results} good results in {search_time:.0f}ms")
            return results, search_time, "balanced", metadata
        
        # Use quality search
        logger.info(f"🔄 Escalating to quality search...")
        results, search_time, metadata = self.search(query_text, top_k, "quality", fetch_metadata)
        good_results = len([r for r in results if r.score > 0.7])
        logger.info(f"✅ Quality search completed: {good_results} good results in {search_time:.0f}ms")
        
        return results, search_time, "quality", metadata
    
    def compare_search_modes(self, query_text, top_k=10, fetch_metadata=True):
        """
        Compare all search modes to help you choose the best one
        """
        logger.info(f"🔍 COMPARING SEARCH MODES for: '{query_text}'")
        
        mode_results = {}
        all_metadata = {}
        
        for mode in ["fast", "balanced", "quality"]:
            logger.info(f"--- {mode.upper()} MODE ---")
            results, search_time, metadata = self.search(query_text, top_k, mode, fetch_metadata)
            
            # Merge metadata
            all_metadata.update(metadata)
            
            # Calculate metrics
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
        
        # Recommend best mode
        best_mode = self._recommend_best_mode(mode_results)
        logger.info(f"🏆 Best mode for this query: **{best_mode.upper()}**")
        
        return mode_results, best_mode, all_metadata
    
    def _recommend_best_mode(self, mode_results):
        """Recommend the best search mode based on results"""
        # Fast mode is good if it has decent results
        if mode_results['fast']['good_results'] >= 3:
            return 'fast'
        
        # Balanced mode if it significantly improves results
        if mode_results['balanced']['good_results'] > mode_results['fast']['good_results'] * 1.2:
            return 'balanced'
        
        # Quality mode for best results
        return 'quality'
    
    def print_results(self, results, search_time, query_text, metadata_dict=None, mode=""):
        """Pretty print search results with metadata"""
        logger.info(f"🔍 SPECTER2 Results for: '{query_text}'")
        logger.info(f"⚡ Search completed in {search_time:.0f}ms {mode}")
        logger.info(f"🚀 Using GPU*{self.gpu_count} optimization")
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
            else:
                logger.info(f"    ⚠️  Metadata not available")