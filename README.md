# Contract Intelligence — Legal Desk

A RAG (Retrieval-Augmented Generation) platform for legal contract analysis. Upload contracts (PDF, DOCX, TXT), ask questions in natural language, and get cited, section-aware answers powered by Gemini.

---


## Architecture

The system uses a 3-phase RAG pipeline per query:

```
User Query
    │
    ▼
Phase 1 ── BERT Clause Classifier
           Classifies query intent into one of 20 clause types
           (e.g. confidentiality, payment_terms, governing_law …)
    │
    ▼
Phase 2 ── Qdrant Vector Search
           Retrieves top-N contract sections via dense embeddings
           (sentence-transformers/all-MiniLM-L6-v2)
    │
    ▼
Phase 3 ── Cross-Encoder Re-ranking
           Re-scores retrieved sections for relevance to the exact query
           (cross-encoder/ms-marco-MiniLM-L-6-v2)
    │
    ▼
Phase 4 ── Gemini LLM Generation
           Generates a cited, section-aware answer with follow-up suggestions
```

Documents are processed on upload:
- Text extraction (PyPDF / Tesseract OCR for scanned PDFs / UnstructuredWord for DOCX)
- Section parsing (headers, clause types, cross-references)
- Embedding generation → stored in Qdrant
- Metadata stored in MongoDB

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Database | MongoDB |
| Vector Store | Qdrant |
| LLM | Google Gemini 2.5 (flash-lite / pro) |
| Embeddings | sentence-transformers all-MiniLM-L6-v2 |
| Clause Classifier | Fine-tuned BERT (bert-base-uncased) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| OCR | Tesseract + Poppler |
| Frontend | React 19 + Vite + Tailwind CSS |

---

## Prerequisites

Install the following before continuing:

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| Node.js | 18+ | For the frontend |
| MongoDB | 6+ | Local or Atlas |
| Qdrant | Any | Local Docker or Qdrant Cloud |
| Tesseract OCR | 5+ | For scanned PDF support |
| Poppler | Latest | Required by pdf2image |
| Gemini API Key | — | From Google AI Studio |



## Project Structure

```
contract-intelligence/
├── backend/
│   ├── src/
│   │   ├── main.py                     # FastAPI app + lifespan startup
│   │   ├── apis/
│   │   │   ├── default.py              # Health check
│   │   │   ├── contractIntelligence.py # Upload & processing endpoints
│   │   │   └── deskConversation.py     # Conversation & RAG endpoints
│   │   ├── configs/
│   │   │   └── config.yaml             # Base config (no secrets)
│   │   ├── services/
│   │   │   ├── section_parser.py       # Splits document into sections
│   │   │   ├── section_embedder.py     # Embeds sections → Qdrant
│   │   │   ├── section_retriever.py    # 3-phase RAG retrieval
│   │   │   ├── crossref_extractor.py   # Detects cross-references between docs
│   │   │   └── entity_extractor.py     # Extracts named entities
│   │   └── utils/
│   │       ├── clause_classifier_instance.py  # BERT classifier singleton
│   │       ├── cross_encoder_instance.py      # Re-ranker singleton
│   │       ├── sentence_transformer_instance.py
│   │       ├── document_utils.py
│   │       ├── pdf_utils.py
│   │       ├── llm_utils.py
│   │       ├── db_utils.py
│   │       ├── connection_utils.py
│   │       ├── log_utils.py
│   │       └── generic_Utils.py        # Config + secrets loader
│   ├── models/
│   │   ├── clause_classifier/          # Fine-tuned BERT (not in git — see below)
│   │   └── section_reranker/           # Cross-encoder weights (not in git — see below)
│   ├── scripts/
│   │   └── download_cross_encoder.py   # One-time model download
│   ├── ci_files/                       # Uploaded contract files (not in git)
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx
    │   ├── api/
    │   ├── components/
    │   └── hooks/
    ├── package.json
    └── vite.config.js
```

---

## Environment Setup

The backend resolves secrets in this priority order:
1. Path set in `ENV_FILE_PATH` environment variable
2. `C:\Users\<YourName>\Desktop\secrets\.env`
3. `.env` in the backend root directory

### Create your `.env` file

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017/contract-intelligence
MONGODB_DB_NAME=contract-intelligence

# Qdrant
QDRANT_CLUSTER_URL=http://localhost
QDRANT_API_KEY=                          # Leave empty for local Qdrant

# Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Google Service Account 
GOOGLE_SERVICE_ACCOUNT_JSON=

# OCR — Windows paths
TESSERACT_CMD_FOR_WINDOWS=C:\Program Files\Tesseract-OCR\tesseract.exe
POPPLER_BIN_PATH_FOR_WINDOWS=C:\Users\YourName\AppData\Local\poppler\poppler-24.08.0\Library\bin

# Security
JWT_SECRET=your_random_secret_here

# Misc
HF_HUB_DISABLE_SYMLINKS_WARNING=1
```

### Environment-specific config overrides (optional)

For environment-specific overrides create `backend/src/configs/config.development.yaml` or `config.production.yaml`. These are excluded from git. Example:

```yaml
# src/configs/config.development.yaml
LLM_PROVIDER: "gemini"
TOTAL_CONCURRENT_LIMIT: 2
```

---

## Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download required ML models (mandatory — run once)
python -m scripts.download_sentence_transformer   # Section embedder (~90 MB)
python -m scripts.download_cross_encoder          # Re-ranker (~87 MB)

python -m scripts.train_clause_classifier             # for local clause classification (optional — requires contracts already uploaded)

```

> The clause classifier is **optional** — if you skip it, the system falls back to LLM-based clause classification automatically. To train it, see [ML Models Setup](#ml-models-setup).

---

## Frontend Setup

```bash
cd frontend

npm install
```

Create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## ML Models Setup

The model weights are **not stored in git** due to size. Follow these steps to set them up.

### 1. Section Embedder (Phase 2)

Downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace into `models/section_embedder/`:

```bash
cd backend
python -m scripts.download_sentence_transformer
```

This only needs to be run once. The server loads from disk on every startup.

**Optional — fine-tune for legal text:**

If you want better retrieval accuracy on your specific contracts, you can fine-tune the embedder using (query, relevant section) pairs from your own data.

Create `backend/data/retrieval_training_data.csv`:

```csv
query,positive
"What is the notice period for termination?","Either party may terminate this Agreement by giving 30 days written notice..."
"Who owns IP created under this agreement?","All intellectual property created by Supplier shall vest in Client..."
```

Tips for building training data:
- 500–2000 pairs is a good starting point
- Queries should look like real user questions, not keyword searches
- Cover all clause types (termination, payment, confidentiality, etc.)
- You can generate synthetic queries from your contracts using an LLM: *"Write 3 questions that this contract section answers: {section_text}"*

Then run:

```bash
cd backend
python -m scripts.train_section_embedder \
    --data_path data/retrieval_training_data.csv \
    --output_dir models/section_embedder \
    --base_model all-MiniLM-L6-v2 \
    --epochs 3 \
    --batch_size 32
```

> **Important after fine-tuning:** The new model produces different vectors. You must re-embed all existing contracts:
> 1. Clear the `contract_sections` Qdrant collection
> 2. Re-upload (or re-process) your contracts via the API

---

### 2. Cross-Encoder Re-ranker (Phase 3)

Downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` (~87 MB) from HuggingFace into `models/section_reranker/`:

```bash
cd backend
python -m scripts.download_cross_encoder
```

This only needs to be run once. After that the model loads from disk on every startup.

---

### 2. Clause Classifier (Phase 1)

This is a fine-tuned BERT model trained to classify contract sections into 20 clause types. You need to train it yourself using your own contract data.

**If you skip this step**, the system still works — it falls back to LLM-based clause classification (slightly slower, no local model needed).

#### Clause types the model classifies

```
definitions, order_of_precedence, term_and_termination, limitation_of_liability,
confidentiality, intellectual_property, governing_law, dispute_resolution,
payment_terms, data_protection, warranties, indemnification, force_majeure,
notices, schedule_or_appendix, business_continuity, audit_rights,
subcontracting, general, other
```

#### Step 1 — Prepare training data

The training script reads directly from your MongoDB `fileSections` collection — no CSV needed. Every contract you upload and process through the system automatically populates this collection with labelled sections (`sectionTitle`, `content`, `clauseType`).

Upload and process at least a few contracts via the UI before training. The more contracts you have processed, the better the model will perform.

#### Step 2 — Train the model

```bash
cd backend
python -m scripts.train_clause_classifier \
    --output_dir models/clause_classifier \
    --base_model bert-base-uncased \
    --epochs 5 \
    --batch_size 16 \
    --max_length 256
```

Training takes ~10–30 minutes on CPU depending on dataset size. A GPU significantly reduces this.

The script saves the final model to `models/clause_classifier/` and a checkpoint to `models/clause_classifier/checkpoint-<N>/`.

#### Step 3 — Verify the model loaded

Restart the backend and look for:

```
INFO - [ClauseClassifier] Loading model from models\clause_classifier\checkpoint-<N> …
INFO - [ClauseClassifier] Model loaded (20 labels).
```

If you see `WARNING - [ClauseClassifier] Model directory not found` instead, the LLM fallback is active — the system still works.

---


## Running the Application

### Backend

```bash
cd backend
python -m src.main
```

Server starts at `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm run dev
```




App opens at `http://localhost:5173`.

---

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/contract-intelligence/upload` | Upload and process a contract file |
| GET | `/api/contract-intelligence/files` | List uploaded files for a conversation |
| DELETE | `/api/contract-intelligence/file/{fileId}` | Delete a file |
| POST | `/api/desk-conversation/ask` | Ask a question (streaming SSE response) |
| GET | `/api/desk-conversation/history` | Get conversation history |
| DELETE | `/api/desk-conversation/conversation` | Clear a conversation |

Full interactive docs available at `http://localhost:8000/docs` when the server is running.

---

