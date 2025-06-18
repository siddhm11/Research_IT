import pandas as pd
from flask import Flask, request, jsonify, render_template, g
from flask_cors import CORS

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchParams

import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

import time
from functools import wraps

# Existing configuration (unchanged)
QDRANT_URL = "https://ba0f9774-1b9e-4b0b-bb05-db8fadfe122c.eu-west-2-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.sMVFQwd_dg3z89uIih5r5olFlbXLAjl_Gcx0V5IJG-U"
COLLECTION_NAME = "arxiv_papers_titles"

SPECTER_COLLECTION = "arxiv_specter2_recommendations"
SPECTER_QDRANT_URL = "https://d09a5111-2452-49a5-b3f8-6a488ca728da.us-east-1-0.aws.cloud.qdrant.io"
SPECTER_QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4xolrSNFliLnWhb7i1Tw1CMbs2pPWJjKu-RgOlQGZTI"

# Initialize models and clients (unchanged)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

app = Flask(__name__)
CORS(app)

# Timing decorator
def timer(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Add timing info to response
        if isinstance(result, tuple):
            response_data, status_code = result
        else:
            response_data = result
            status_code = 200
            
        if hasattr(response_data, 'get_json'):
            json_data = response_data.get_json()
            json_data['execution_time_ms'] = round(execution_time, 2)
            return jsonify(json_data), status_code
        
        return result
    return wrapper

def setup_specter_model():
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
        model.half()
    return model, tokenizer

specter_model, specter_tokenizer = setup_specter_model()
specter_client = QdrantClient(url=SPECTER_QDRANT_URL, api_key=SPECTER_QDRANT_API_KEY, timeout=120)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_similar', methods=['POST'])
@timer
def find_similar():
    start_time = time.perf_counter()
    
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = data.get("top_k", 100)  # Default to 100

    if not query_text.strip():
        return jsonify({"error": "Missing or empty query_text"}), 400

    # Embedding generation timing
    embed_start = time.perf_counter()
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0].astype(np.float16)
    embed_time = (time.perf_counter() - embed_start) * 1000

    # Search timing
    search_start = time.perf_counter()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(exact=False)
    )
    search_time = (time.perf_counter() - search_start) * 1000

    response = {
        "model": "MiniLM",
        "results": [
            {
                "arxiv_id": r.payload.get("arxiv_id", "N/A"),
                "score": r.score
            }
            for r in results
        ],
        "timing": {
            "embedding_ms": round(embed_time, 2),
            "search_ms": round(search_time, 2),
            "total_ms": round((time.perf_counter() - start_time) * 1000, 2)
        },
        "total_results": len(results)
    }
    return jsonify(response)

@app.route('/specter_search', methods=['POST'])
@timer
def specter_search():
    start_time = time.perf_counter()
    
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = data.get("top_k", 100)  # Default to 100

    if not query_text.strip():
        return jsonify({"error": "Missing or empty query_text"}), 400

    # Tokenization timing
    tokenize_start = time.perf_counter()
    inputs = specter_tokenizer(
        query_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        return_token_type_ids=False
    )
    tokenize_time = (time.perf_counter() - tokenize_start) * 1000

    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # Model inference timing
    inference_start = time.perf_counter()
    with torch.no_grad():
        outputs = specter_model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].astype(np.float16)
    inference_time = (time.perf_counter() - inference_start) * 1000

    # Search timing
    search_start = time.perf_counter()
    results = specter_client.search(
        collection_name=SPECTER_COLLECTION,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )
    search_time = (time.perf_counter() - search_start) * 1000

    response = {
        "model": "SPECTER2",
        "results": [
            {
                "arxiv_id": r.payload.get("arxiv_id", "N/A"),
                "score": r.score
            }
            for r in results
        ],
        "timing": {
            "tokenization_ms": round(tokenize_time, 2),
            "inference_ms": round(inference_time, 2),
            "search_ms": round(search_time, 2),
            "total_ms": round((time.perf_counter() - start_time) * 1000, 2)
        },
        "total_results": len(results)
    }
    return jsonify(response)

# New endpoint to compare both models
@app.route('/compare_models', methods=['POST'])
@timer
def compare_models():
    start_time = time.perf_counter()
    
    data = request.get_json()
    query_text = data.get("query_text", "")
    top_k = data.get("top_k", 100)

    if not query_text.strip():
        return jsonify({"error": "Missing or empty query_text"}), 400

    # Get results from both models
    minilm_start = time.perf_counter()
    
    # MiniLM search
    query_embedding_minilm = model.encode([query_text], normalize_embeddings=True)[0].astype(np.float16)
    minilm_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding_minilm.tolist(),
        limit=top_k,
        with_payload=True,
        search_params=SearchParams(exact=False)
    )
    minilm_time = (time.perf_counter() - minilm_start) * 1000

    # SPECTER2 search
    specter_start = time.perf_counter()
    inputs = specter_tokenizer(
        query_text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
        return_token_type_ids=False
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = specter_model(**inputs)
        query_embedding_specter = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0].astype(np.float16)

    specter_results = specter_client.search(
        collection_name=SPECTER_COLLECTION,
        query_vector=query_embedding_specter.tolist(),
        limit=top_k,
        with_payload=True
    )
    specter_time = (time.perf_counter() - specter_start) * 1000

    # Calculate similarity between results
    minilm_arxiv_ids = [r.payload.get("arxiv_id") for r in minilm_results]
    specter_arxiv_ids = [r.payload.get("arxiv_id") for r in specter_results]
    
    # Intersection-based similarity calculation
    common_papers = set(minilm_arxiv_ids).intersection(set(specter_arxiv_ids))
    similarity_percentage = (len(common_papers) / top_k) * 100

    # Find overlap at different top-k values
    overlap_analysis = {}
    for k in [10, 25, 50, 100]:
        if k <= top_k:
            minilm_top_k = set(minilm_arxiv_ids[:k])
            specter_top_k = set(specter_arxiv_ids[:k])
            overlap = len(minilm_top_k.intersection(specter_top_k))
            overlap_analysis[f"top_{k}"] = {
                "overlap_count": overlap,
                "overlap_percentage": round((overlap / k) * 100, 2)
            }

    response = {
        "query": query_text,
        "minilm_results": [
            {"arxiv_id": r.payload.get("arxiv_id", "N/A"), "score": r.score}
            for r in minilm_results
        ],
        "specter_results": [
            {"arxiv_id": r.payload.get("arxiv_id", "N/A"), "score": r.score}
            for r in specter_results
        ],
        "similarity_analysis": {
            "common_papers": list(common_papers),
            "total_common": len(common_papers),
            "similarity_percentage": round(similarity_percentage, 2),
            "overlap_analysis": overlap_analysis
        },
        "timing": {
            "minilm_ms": round(minilm_time, 2),
            "specter_ms": round(specter_time, 2),
            "total_ms": round((time.perf_counter() - start_time) * 1000, 2)
        }
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
