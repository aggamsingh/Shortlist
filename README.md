# Resume Screening Microservice

A production-ready, self-contained, CPU-only resume screening microservice. It implements a Retrieval-Augmented Generation (RAG) pipeline to fetch matching resume chunks from a Qdrant vector database and rerank them using Groq or Gemini API.

## Features
- **Incremental Indexing**: Skips unmodified resumes using MD5 hashing (saved in `./data/index_state.json`).
- **Rich Document Parsing**: Supports both PDF and DOCX CVs (including tables parsing).
- **CPU-Optimized**: Uses a CPU-only PyTorch build to keep Docker image size small (~80MB embedding model).
- **Robust Reranking**: Intelligently groups chunks, builds candidate summaries, and queries Gemini/Groq under JSON schemas.

---

## 🛠 Setup & Installation

### 1. Place Resumes
Create a directory named `cvs` in the root of the project folder (if it doesn't already exist), and place all candidate CV files (`.pdf` and `.docx`) inside:
```bash
mkdir cvs
# copy your CVs into the cvs folder
```

### 2. Configure Environment Variables
Copy the configuration template:
```bash
cp .env.example .env
```
Open the new `.env` file and set:
- **`GROQ_API_KEY`** or **`GEMINI_API_KEY`**: Provide at least one API key to perform candidate reranking. (If both are set, Gemini is used by default).
- **`API_KEY`**: Your internal shared API key used to secure the service. Example: `API_KEY=my_secure_handshake_key`
- **`CV_FOLDER_PATH`**: Host path pointing to your CV storage folder (defaults to `./cvs`).

---

## 🚀 Running the Service

### 1. Build and Run the Vector DB and API Server
Spin up Qdrant and the FastAPI API server to run 24/7 in the background:
```bash
docker compose up -d
```
Verify the server is running by viewing the logs:
```bash
docker compose logs -f app
```

### 2. Run the Resume Indexer
Whenever you add new CVs or update existing ones, trigger the indexer script to index the resumes:
```bash
docker compose run indexer
```
*Note: Thanks to hashing logic, subsequent index runs only process new or modified files, preserving bandwidth and CPU resources.*

### 3. Stop the Service
To temporarily stop the database and API server:
```bash
docker compose down
```

---

## 📡 API Endpoints

All client requests must include the header: `X-API-Key: <your_internal_api_key_here>`.

### 1. Screen Resumes
**Endpoint:** `POST /api/v1/screen`

**Request Headers:**
```http
X-API-Key: your_internal_api_key_here
Content-Type: application/json
```

**Request Body (JSON):**
```json
{
  "job_description": "We are seeking a Backend Software Engineer with 3+ years of experience in Python, FastAPI, Docker, and Qdrant. The engineer will build microservices and vector search components. Based in Delhi.",
  "top_k": 5,
  "filters": {
    "min_experience": 3,
    "location": "Delhi"
  }
}
```

**Example Curl Command:**
```bash
curl -X POST http://localhost:8000/api/v1/screen \
  -H "X-API-Key: your_internal_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Seeking Python engineer with 3+ years experience, based in Delhi",
    "top_k": 5,
    "filters": {
      "min_experience": 3,
      "location": "Delhi"
    }
  }'
```

**Response Body (JSON):**
```json
{
  "job_id": "90ba9535-90eb-4fb6-ba64-58a69d2f2d9c",
  "candidates": [
    {
      "candidate_id": "d3b07384-d113-5f8a-9293-1829e34a2e58",
      "name": "Jane Doe",
      "score": 0.94,
      "match_reasoning": "Strong match with 4 years experience in Python and FastAPI, located in Delhi.",
      "cv_path": "/app/cvs/Jane_Doe_CV.pdf"
    }
  ],
  "screened_at": "2026-07-11T12:00:00Z"
}
```

### 2. Service Health Check
**Endpoint:** `GET /health`

**Example Curl Command:**
```bash
curl http://localhost:8000/health
```

**Response Body (JSON):**
```json
{
  "status": "healthy",
  "qdrant_connected": true,
  "model_loaded": true,
  "llm_configured": true
}
```

---

## 📂 Project Structure
```
resume-screener/
├── docker-compose.yml     # Compose file orchestrating services
├── Dockerfile             # Core python image configuration
├── .env.example           # Config file template
├── requirements.txt       # Core dependencies (CPU-only PyTorch)
├── indexer/
│   ├── run.py             # Main indexing script
│   ├── parser.py          # CV parsers (PDF, DOCX) & heading chunker
│   ├── embedder.py        # SentenceTransformer CPU singleton
│   └── utils.py           # Logging, hashing, and state file utils
├── api/
│   ├── main.py            # FastAPI endpoints and lifespans
│   ├── retriever.py       # Qdrant queries & candidate deduplication
│   ├── reranker.py        # Groq/Gemini JSON mode call logic
│   └── models.py          # Pydantic request/response schemas
├── cvs/                   # Folder where resumes are dropped
└── data/                  # Persistent data folder (caching models and hashes)
```

## 🔧 Troubleshooting

- **Hugging Face model downloads on restarts**: The system mounts `/app/data` to preserve the `hf_home` directory. The embedding model is cached there during the very first startup or run and is reused afterwards, saving network load.
- **Out of Memory / High CPU during Indexing**: The indexer processes resumes one-by-one and chunks them in batches. Ensure the host system has at least 2GB of free RAM.
- **Port Conflicts**: If port `6333` (Qdrant) or `8000` (FastAPI) is occupied by other software on your server, change them in your `.env` file.
