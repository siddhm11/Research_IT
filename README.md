# ResearchIT Backend — Personalized Paper Recommendation API

**A FastAPI-powered backend for academic paper search and personalized recommendations, leveraging SPECTER2 and MiniLM embeddings, Qdrant vector storage, and dynamic user modeling with MMR diversification.**

---

## 🚀 Core Features

- **Dual-Model Semantic Search**
  - **SPECTER2** — deep semantic search over paper *abstracts* for high-quality relevance.
  - **MiniLM** — lightweight, fast title-based search for quick lookups.
  - **Auto-Search** — system chooses SPECTER2 or MiniLM per-query for best tradeoff between speed and quality.

- **Deep User Personalization**
  - **Multi-vector profiles** — each user stores four vectors in Qdrant: one complete profile + up to three subject-specific profiles.
  - **Temporal decay** — interaction signals fade with time so recommendations track current interests.
  - **Interaction tracking** — model learns from `view`, `like`, `bookmark`, `dislike` and updates user vectors continuously.

- **Personalized & Diverse Feeds**
  - **MMR-based feed** — Maximal Marginal Relevance to balance relevance and diversity; user-configurable `mmr_lambda`.
  - **Efficient metadata retrieval** — only requested metadata is paginated/fetched to minimize overhead.

- **System Management**
  - Health checks, system statistics, and maintenance utilities (cleanup, reindexing, vector tuning).

---

## 🛠 Tech Stack

- **Framework**: FastAPI
- **Server**: Uvicorn
- **Vector DB**: Qdrant
- **Embeddings**:
  - SPECTER2 (`allenai/specter2_base`)
  - MiniLM (`sentence-transformers/all-MiniLM-L6-v2`)
- **Database**: SQLite (lightweight user mappings) — replaceable with Postgres for production
- **Metadata Source**: arXiv API
- **Deployment**: Docker / Docker Compose (recommended for production parity)

---

## ⚙️ Quickstart — Local Development

1. **Clone**

```bash
git clone <your-repo-url>
cd backend
```

2. **Virtual environment & install**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure env**

> Move secrets from code to env or `.env`. Never commit them.

Required environment variables (examples):

- `QDRANT_URL` — Qdrant URL for full-profile vectors
- `QDRANT_TITLE_URL` — Qdrant URL for title/MiniLM vectors (if separate)
- `SQLITE_PATH` — path to SQLite DB
- `ARXIV_API_URL` — (optional) arXiv endpoint
- `SPECTER2_MODEL_PATH` — local path / HF model id if caching

4. **Run**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open: `http://localhost:8000`

---

## 🗺 API Reference (selected)

### Search

- `POST /search` — semantic search (SPECTER2)
  - Params: `query`, `count`, `search_mode` (`fast|balanced|quality`)

- `POST /title-search` — title-only search (MiniLM)

- `POST /auto-search` — choose between SPECTER2 and MiniLM automatically

- `POST /similar` — similar papers by `arxiv_id`

- `POST /personalized-search` — blend `query` with user profile vectors


### User Management

- `POST /users` — create user profile
- `GET /users/{user_id}` — fetch user stats/profile
- `POST /users/{user_id}/onboard` — seed subjects and liked papers
- `POST /users/{user_id}/interact` — record interactions `{"action":"like","paper_id":"..."}`

### Personalized Feed

- `GET /users/{user_id}/feed` — paginated personalized feed
- `GET /users/{user_id}/feed/multi-vector` — merged feed from complete + subject vectors
- `POST /users/{user_id}/mmr-config` — set `mmr_lambda` (0..1)

### System

- `GET /health`
- `GET /stats`
- `GET /capabilities`

---

## 🧠 Personalization: details

### User profile vectors
- **Structure**: 4 vectors stored per user in Qdrant: `complete`, `subject1`, `subject2`, `subject3`.
- **Updates**: on every interaction we compute embedding(s) for the referenced paper and update profile vectors using a weighted sum + decay.

### Temporal decay

We apply exponential decay to older interactions:

\[
w_t = w_0 \cdot e^{-\lambda t}
\]

- `w_0` — base action weight (e.g., like=3, view=1, bookmark=4, dislike=-2)
- `t` — time since action (seconds / days)
- `λ` — decay tuned via `half_life` config

### MMR (Maximal Marginal Relevance)

To return a set `S` of documents that are both relevant and diverse, we score candidates with:

\[
\text{score}(d_i) = \lambda \cdot \text{sim}(q, d_i) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)
\]

- `λ` (`mmr_lambda`) controls relevance vs diversity. Lower `λ` → more diverse results.

---

## 🔁 Example Workflows

1. **Create a user**

```bash
curl -X POST http://localhost:8000/users
```

2. **Onboard user**

```bash
curl -X POST http://localhost:8000/users/{user_id}/onboard \
  -H "Content-Type: application/json" \
  -d '{"subject1_name":"AI","subject2_name":"NLP","subject3_name":"CV","liked_papers":["xyz123"]}'
```

3. **Personalized search**

```bash
curl -X POST http://localhost:8000/personalized-search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"{user_id}","query":"transformers","count":10}'
```

4. **Get feed**

```bash
curl http://localhost:8000/users/{user_id}/feed?count=10&page=1
```

5. **Adjust MMR**

```bash
curl -X POST http://localhost:8000/users/{user_id}/mmr-config \
  -H "Content-Type: application/json" \
  -d '{"mmr_lambda":0.3}'
```

---

## 📚 References

- AllenAI SPECTER2 — https://github.com/allenai/specter2
- Sentence Transformers / MiniLM — https://www.sbert.net/
- Qdrant — https://qdrant.tech/
- arXiv API — https://arxiv.org/help/api/user-manual

---

## 🛡 Security & Ops

- **Secrets**: keep Qdrant credentials and API keys in env or secret manager.
- **Scaling**: move from SQLite → Postgres; shard or partition Qdrant collections for scale.
- **Vector tuning**: tune `M` and `ef_search` for ANN indices; use quantization to reduce memory.
- **Observability**: expose Prometheus metrics, traces, and health endpoints.

---

## 🧩 Troubleshooting & Tips

- If feeds are sparse: verify onboarding seeds and decay settings.
- If search is slow: verify Qdrant network latency, tune `ef` and consider separate collections for titles vs abstracts.
- If relevance is poor: evaluate embedding model fingerprints and consider reindexing with metadata boosts (year, citations).

---

## ❤️ Contributions

PRs, issues and ideas welcome. Please document any new endpoints, schema changes, or operational steps in the repo.

---
