# DocuMind — Multi-Document RAG with Conversational Memory

> Ask questions across your PDFs. Get concise answers grounded in retrieved passages, with document and page references.

DocuMind is a full-stack document question-answering application. A user creates a conversation, uploads one or more PDFs into that conversation, and asks natural-language questions. The backend extracts and chunks the PDF text, creates embeddings, stores vectors in a session-specific FAISS index, retrieves relevant passages, and asks a Groq-hosted language model to answer using only that retrieved evidence. Answers include source labels that map back to document names and PDF pages.

**Project type:** Machine Learning / Generative AI engineering assessment  
**Backend:** Python, FastAPI, LangChain, FAISS, SQLite  
**Frontend:** React, TypeScript, Vite  
**Embedding model:** `BAAI/bge-small-en-v1.5` (configured for CPU)  
**Generation model:** Groq `openai/gpt-oss-20b` (configurable)

---

## 1. The problem we are solving

Reading multiple long PDFs and manually locating facts is slow. A conventional chatbot may also answer from general model knowledge instead of the supplied documents, making it difficult to verify the result.

DocuMind addresses this by combining:

1. **Document ingestion** — extract readable text and preserve page-level metadata.
2. **Semantic retrieval** — find passages related to a user's question, even when the wording differs.
3. **Evidence-grounded generation** — provide the retrieved passages to the language model as its factual context.
4. **Source attribution** — attach document/page references to the answer.
5. **Conversational memory** — resolve short follow-up questions using recent turns from the same conversation.
6. **Session isolation** — keep each conversation's documents and vector index separate.

The design goal is not to make the language model “know” every PDF. The goal is to retrieve the relevant evidence first, then use the model to express that evidence clearly.

## 2. Product workflow at a glance

1. The React application checks the API health and loads or creates a conversation.
2. The user uploads PDF files to that conversation.
3. FastAPI validates the files and passes them to the document service.
4. The loader extracts text page by page. Page number and source filename are retained as metadata.
5. The text splitter creates overlapping chunks.
6. The embedding model converts each chunk into a numerical vector.
7. FAISS stores those vectors in the index belonging to that conversation.
8. When the user asks a question, the backend optionally rewrites a context-dependent follow-up into a standalone search question.
9. The retriever searches the active conversation's FAISS index and expands selected chunks with nearby chunks from the same source page.
10. The RAG chain builds a bounded context with source labels such as `[S1]`.
11. Groq generates an answer constrained to the retrieved evidence.
12. The backend validates citation labels, returns the answer and cited sources, and persists the turn in SQLite.
13. React renders the answer and its source chips.

See [the architecture guide](docs/ARCHITECTURE.md) for the component diagram and data flow.

## 3. Why each technology is used

| Component | Technology | Why it is used here |
|---|---|---|
| User interface | React + TypeScript | Component-based UI with explicit data types for sessions, documents, messages and sources. |
| Frontend tooling | Vite | Fast local development server and production build pipeline. |
| API layer | FastAPI | Typed request/response models, HTTP endpoints, upload handling and interactive OpenAPI docs. |
| PDF parsing | PyMuPDF | Extracts text page by page and allows page metadata to be retained for citations. |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits long page text into retrieval-sized passages while preferring natural text boundaries. |
| Embeddings | `BAAI/bge-small-en-v1.5` | Converts text and queries into dense vectors so semantically related passages can be found. |
| Vector search | FAISS | Efficient local similarity search without requiring a hosted vector database. |
| LLM orchestration | LangChain | Connects prompts, model calls and output parsing into a readable RAG pipeline. |
| Answer generation | Groq `openai/gpt-oss-20b` | Generates concise natural-language responses from the retrieved context. |
| Session and chat persistence | SQLite | Stores conversations, messages, source metadata and uploaded-document records locally. |

### Why local FAISS instead of a hosted vector database?

The assessment prototype is designed to run locally and avoid requiring a separately managed vector database service. FAISS provides vector indexing and similarity search within the Python application. Each session has its own index directory, which supports the document-isolation requirement.

This is a local persistence design, not a claim that FAISS automatically provides multi-machine replication, distributed search, or production-grade concurrent index management.

### Why use embeddings instead of keyword matching?

A keyword search can miss relevant passages when a question uses different wording from the source. Embeddings represent text in a vector space and allow semantic similarity search. For example, a question asking about “annual paid time off” may retrieve a passage phrased as “paid annual leave”.

Semantic similarity is a retrieval signal, not proof that a passage answers the question. The generation prompt and citation/abstention checks are additional safeguards.

## 4. Ingestion pipeline — what happens after a PDF is uploaded?

### Step 1: Validate the upload

The document service:
- accepts PDF files only;
- rejects empty files;
- enforces a 20 MiB per-file limit;
- checks for a PDF header;
- attempts to parse the file and rejects unreadable PDFs;
- sanitizes the display filename instead of trusting client-supplied path components.

**Why:** Validation prevents unsupported or malformed files from entering the indexing pipeline and constrains upload size.

### Step 2: Extract text page by page

`PDFDocumentLoader` uses PyMuPDF to read each page. It skips empty pages and removes known repeated synthetic footer text and standalone page-number lines. Every extracted page is represented as a LangChain `Document` with metadata including:
- `source` — original filename;
- `page` — one-based PDF page number;
- `file_path` — stored file path.

**Why:** Page-level extraction allows the retriever and answer layer to identify where evidence came from. Cleaning repeated footer/page-number noise reduces irrelevant text in chunks.

### Step 3: Split text into chunks

The configured splitter uses:
- `CHUNK_SIZE = 350`
- `CHUNK_OVERLAP = 70`
- separators that prefer paragraph breaks, line breaks, sentence boundaries, then spaces.

The splitter also records a start index and assigns chunk IDs.

**Why:** Embedding an entire long PDF page or document as one vector can blur multiple topics into one representation. Smaller passages provide more focused retrieval units. Overlap preserves some context across chunk boundaries so a fact split between adjacent chunks is less likely to be lost.

These values are configuration choices for this prototype, not universally optimal values. Different document types may require tuning.

### Step 4: Create embeddings

`BAAI/bge-small-en-v1.5` is loaded with the configured device (`cpu`) and normalized embeddings. The query path prepends the configured search instruction:

`Represent this sentence for searching relevant passages: `

The embedding provider is process-wide and lazy: the first request that needs embeddings initializes the model, and subsequent FAISS wrappers in the same Python process reuse that instance.

**Why:** A shared instance avoids repeatedly loading the same model during index creation, index loading and retrieval. It reduces redundant initialization overhead; no numerical latency or memory improvement is claimed because a benchmark has not been run.

### Step 5: Persist a session-specific FAISS index

Uploaded PDFs are stored under a session-specific upload directory. The index is stored under a session-specific vector-store directory. The document service rebuilds that session's index from its registered PDFs.

The rebuild process writes to a temporary sibling directory and swaps it into place only after index creation succeeds. If the swap fails, the previous index is restored where possible.

**Why:** Indexing by session prevents one conversation from searching another conversation's files. Rebuilding from the current registered-document list ensures a deleted PDF is not retained as stale vectors.

## 5. Question-answering pipeline — how does a response get generated?

This is the core RAG flow.

### Step 1: Receive the question and recent conversation history

The frontend sends the question to the session-specific chat endpoint. The backend loads up to the most recent 10 messages for that session.

**Why:** A follow-up such as “And what about the previous year?” is ambiguous without its preceding question.

### Step 2: Rewrite a follow-up into a standalone question

When history exists, a dedicated rewrite prompt asks the LLM to resolve references such as “it”, “that”, or “the previous year”. It explicitly instructs the model to preserve the requested entity and metric and to change only the relevant period when appropriate.

Example:

- Previous user question: “What was the company's total revenue in FY 2025-26?”
- Follow-up: “And what about the previous financial year?”
- Resolved search question: “What was the company's total revenue in FY 2024-25?”

If the latest question is already standalone, the rewrite prompt is instructed to return it unchanged.

**Why:** Retrieval works better with a self-contained query. Scope-preserving instructions also reduce the risk that a narrow follow-up expands into unrelated metrics from the conversation.

### Step 3: Retrieve relevant chunks from the active session

The query is embedded using the same embedding model and searched against that session's FAISS index. The configured `RETRIEVAL_TOP_K` is 5.

`ContextualRetriever` then expands selected chunks by a neighbor window of 1, but only within the same source document and PDF page. When adjacent chunks overlap, the retriever attempts to join them without duplicating the shared text.

**Why:** The best-matching chunk may contain only half of a sentence, table row or explanation. Including adjacent chunks can restore enough local context to interpret the passage. The same-page and same-source constraints avoid joining unrelated documents.

### Step 4: Build bounded evidence context and source labels

The RAG chain prepares context from the retrieved documents and assigns labels such as:

`[S1] Source: annual_report.pdf, Page: 2`

The context is capped at 10,000 characters.

**Why:** Source labels let the model refer to evidence in a predictable format. A context limit controls how much retrieved text is sent to the generation model and helps keep the prompt focused.

### Step 5: Generate an answer from evidence

The answer prompt tells the LLM:
- retrieved excerpts are the only factual evidence;
- conversation history is not evidence;
- it must answer only the resolved question;
- it must not invent facts, numbers or policies;
- unsupported questions must receive an explicit “not found” response;
- factual claims must cite the supplied source labels.

The configured Groq model uses temperature `0`.

**Why:** The model is used to interpret and communicate retrieved passages, not to fill missing facts from its general knowledge. Temperature zero makes generation settings more deterministic, but does not guarantee factual correctness.

### Step 6: Validate and return citations

The backend normalizes common citation bracket variants, removes source labels that were not supplied to the model, and checks whether the answer contains at least one valid citation. If retrieval is empty, context is unusable, or the generated answer has no valid citation, the chain returns a safe abstention response and no cited sources.

Only source references actually cited in the answer are returned to the frontend. The session manager persists assistant messages with their source metadata.

**Important limitation:** This is a citation-format and source-ID validation guard. It does not mathematically prove that every cited passage entails every sentence. Human review and stronger automated evaluation are still useful for high-stakes use cases.

## 6. How answer quality and reliability were improved

DocuMind uses several complementary controls. No measured accuracy percentage is claimed.

| Design improvement | Failure mode it is intended to reduce | How it works |
|---|---|---|
| Page-level metadata | Answers without traceable origin | Preserves source filename and page number from PDF extraction through retrieval. |
| Chunk overlap | Facts split at chunk boundaries | Adjacent chunks share 70 characters. |
| Contextual neighbor retrieval | Retrieved snippet lacks surrounding explanation | Expands a matched chunk by one neighboring chunk on either side, within the same source page. |
| Follow-up question rewriting | Ambiguous pronouns and relative periods | Converts context-dependent messages into standalone search queries. |
| Scope-preserving rewrite prompt | Follow-up unexpectedly changes the metric or broadens scope | Instructs the rewrite model to preserve subject and metric. |
| Focused answer prompt | Model adds unrelated facts from context or prior conversation | Sends the resolved question and retrieved excerpts, not the full chat history, to answer generation. |
| Evidence-only instructions | Model fills gaps from prior knowledge | Requires the answer to rely only on retrieved excerpts. |
| Citation normalization and ID filtering | Malformed or invented source labels | Normalizes common bracket forms and removes unknown source IDs. |
| Citation-required response gate | Uncited generated answer appears authoritative | Abstains if no valid source ID is cited. |
| Empty-retrieval abstention | Model answers when nothing relevant was retrieved | Returns a not-found response without invoking answer generation. |
| Session-specific indexes | Cross-conversation document leakage | Each session uploads, indexes, searches and deletes its own documents. |

### What “accuracy improvement” means in this project

The project improves answer reliability through retrieval context, scope control, traceability and abstention. These are engineering safeguards, not a statistically measured accuracy score.

A rigorous accuracy claim would require a labelled evaluation set with expected answers and supporting passages, then metrics such as retrieval Recall@K, MRR/nDCG, answer correctness, citation precision/recall and abstention quality. Those benchmark results are not currently included in this repository.

## 7. Conversational memory and session isolation

### Conversational memory

SQLite stores sessions and ordered messages. Before a chat request, the backend formats up to 10 recent messages as conversation history. That history is used for follow-up rewriting.

The answer-generation step receives the standalone resolved question and retrieved document context. It does not receive the full conversation history as factual evidence.

**Why this separation matters:** History helps interpret the user's intent; uploaded documents support factual answers. Keeping those roles separate reduces the chance of answering from an earlier assistant response rather than from the source PDFs.

### Session isolation

A session owns:
- its SQLite session record and messages;
- its registered document records;
- its uploaded PDF directory;
- its FAISS index directory;
- its cached RAG chain while the API process is running.

When a document is deleted, its record and file are removed and that session's index is rebuilt from the remaining registered documents. When a session is deleted, the application removes its upload and index directories and deletes its SQLite session record (with related rows configured for cascading deletion).

## 8. API overview

The FastAPI service exposes the following routes:

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/sessions` | List conversations |
| `POST` | `/api/sessions` | Create a conversation |
| `GET` | `/api/sessions/{session_id}` | Load messages and documents |
| `DELETE` | `/api/sessions/{session_id}` | Delete a conversation and its stored files/index |
| `GET` | `/api/sessions/{session_id}/documents` | List documents in a conversation |
| `POST` | `/api/sessions/{session_id}/documents` | Upload and index PDFs |
| `DELETE` | `/api/sessions/{session_id}/documents/{document_id}` | Delete a PDF and rebuild that session index |
| `POST` | `/api/sessions/{session_id}/chat` | Ask a question and receive answer, sources and resolved question |

Interactive API documentation is available at `/docs` while the backend is running.

## 9. Repository structure

The source archive supplied for documentation contained this structure. The optimized embedding implementation adds `backend/tests/test_embedding_model.py` and updates the embedding and FAISS modules.

```text
DocuMind/
├── README.md
├── .gitignore
├── docs/
│   ├── ARCHITECTURE.md
│   └── DEMO_SCRIPT.md
├── backend/
│   ├── .env.example
│   ├── .gitignore
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/
│   │   │   └── main.py
│   │   ├── chains/
│   │   │   ├── chat_history.py
│   │   │   ├── llm.py
│   │   │   └── rag_chain.py
│   │   ├── embeddings/
│   │   │   └── embedding_model.py
│   │   ├── ingestion/
│   │   │   ├── document_loader.py
│   │   │   └── text_splitter.py
│   │   ├── memory/
│   │   │   └── session_manager.py
│   │   ├── retrieval/
│   │   │   └── retriever.py
│   │   ├── services/
│   │   │   └── document_service.py
│   │   └── vectorstore/
│   │       └── faiss_store.py
│   ├── data/
│   │   └── sample_documents/
│   └── tests/
└── frontend/
    ├── .env.example
    ├── package.json
    └── src/
        ├── App.tsx
        ├── api.ts
        ├── main.tsx
        ├── styles.css
        └── types.ts
```

Runtime files such as SQLite databases, uploaded user PDFs and generated FAISS indexes are created under `backend/data/`; they should not be committed.

## 10. Local setup

### Prerequisites

- Python 3.11 recommended (use the Python version already validated with this project).
- Node.js and npm compatible with the Vite project.
- A Groq API key for answer generation.
- Internet access for the first embedding-model download and Groq API calls.
- CPU execution is configured by default; a GPU is not required.

### 10.1 Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DocuMind
```

### 10.2 Configure the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create `backend/.env` from `backend/.env.example` and set your Groq key:

```dotenv
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu
```

Do not commit `.env` or paste its contents into issues, screenshots or public messages.

Start the API from the `backend` directory:

```powershell
uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
```

Useful URLs:
- API health: `http://127.0.0.1:8000/api/health`
- Swagger docs: `http://127.0.0.1:8000/docs`

### 10.3 Configure the frontend

Open another terminal at the repository root:

```powershell
cd frontend
npm install
```

The frontend reads `VITE_API_BASE_URL` and defaults to `http://127.0.0.1:8000/api`. The example file is `frontend/.env.example`. Copy it to `frontend/.env` only if you need to override the default.

Start the frontend:

```powershell
npm run dev
```

Open `http://127.0.0.1:5173`.

### 10.4 First-run model download

On first use, the Hugging Face embedding model may be downloaded and cached locally. Later starts can reuse the local cache. The embedding model is loaded lazily by the backend process and shared by its FAISS wrappers.

## 11. Configuration reference

The following defaults are defined in `backend/app/config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `APP_NAME` | `DocuMind` | Application display/service name |
| `APP_VERSION` | `1.0.0` | API version string |
| `GROQ_API_KEY` | empty | Secret key required for Groq |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Generation model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence embedding model |
| `EMBEDDING_DEVICE` | `cpu` | Embedding execution device |
| `CHUNK_SIZE` | `350` | Chunk size in characters |
| `CHUNK_OVERLAP` | `70` | Overlap in characters |
| `RETRIEVAL_TOP_K` | `5` | Number of initial FAISS matches |
| `DATA_DIR` | `data` | Runtime data directory |
| `VECTOR_DB_DIR` | `data/vectorstore` | Root for per-session indexes |

The RAG chain currently also uses a one-chunk neighbor window and a 10,000-character context limit.

## 12. Testing and verification

Run commands from `backend` with the project virtual environment active.

### Focused automated tests

```powershell
python -m unittest tests.test_embedding_model -v
python -m unittest tests.test_rag_quality -v
```

The embedding test verifies that the `EmbeddingModel` wrapper and multiple FAISS stores reuse the same initialized embedding object in one process.

The RAG-quality tests cover:
- empty retrieval abstention;
- using the resolved standalone question for answer generation;
- removing unknown citations and abstaining from uncited answers;
- returning the source corresponding to a valid citation.

### Integration-style scripts

```powershell
python -m tests.test_rag
python -m tests.test_memory
python -m tests.test_document_isolation
```

These scripts exercise the RAG path, follow-up memory behavior and per-session document isolation/deletion. Some integration scripts may require a configured Groq key and a locally available embedding model.

Other test modules present in the repository include ingestion, LLM and vector-store tests. Run them individually when their external model/API dependencies are available:

```powershell
python -m unittest tests.test_ingestion -v
python -m unittest tests.test_llm -v
python -m unittest tests.test_vectorstore -v
python -m unittest tests.test_vectorstore_contextual -v
```

### Verified project behavior

During the developer's local manual verification, the following were reported as passing:
- backend and frontend startup;
- upload of the three supplied sample PDFs;
- source-grounded answers for known financial, HR and technical facts;
- citations linked to source documents/pages;
- follow-up question resolution;
- session isolation;
- document and session deletion cleanup;
- abstention for information not found in the uploaded documents;
- no critical frontend console errors.

Automated test output previously shared by the developer showed:
- `test_embedding_model`: 1 test passed;
- `test_rag_quality`: 4 tests passed;
- `test_document_isolation`: session isolation and document deletion checks passed;
- RAG and conversational-memory integration scripts also produced expected answers in earlier runs.

These are local development test results, not an independent third-party audit or production load/security certification.

## 13. Sample evaluation documents

The repository includes three synthetic sample PDFs:
- `01_annual_financial_report.pdf`
- `02_technical_architecture_report.pdf`
- `03_hr_and_operational_policy.pdf`

They describe a fictional organization and are intended only for demonstrating ingestion, retrieval, citations and follow-up questions. They are not real company records or externally verified facts.

Example questions for the demo:
- What was total revenue in FY 2025-26?
- And what about the previous financial year?
- What was net profit in FY 2025-26?
- How many paid annual leave days are provided?
- How many days per week can employees work remotely?
- What is the target API availability?
- Does the uploaded policy specify parental leave entitlement?

## 14. Limitations and future improvements

This is an assessment-ready local prototype, not yet a hardened multi-tenant production service.

Known boundaries and sensible next steps:
- **Scanned PDFs:** Current extraction uses PDF text extraction. Scanned image-only PDFs may need OCR.
- **Evaluation metrics:** Add a labelled benchmark and report retrieval/answer/citation metrics before claiming a numeric accuracy improvement.
- **Reranking:** Add a cross-encoder reranker if evaluation shows top-k retrieval needs improvement.
- **Chunking strategy:** Evaluate chunk size/overlap against document types such as tables, policies and technical reports.
- **Citation entailment:** Current validation checks citation IDs and presence, not full semantic entailment.
- **Authentication and authorization:** Add user identity, access control and tenant boundaries before exposing the service publicly.
- **Rate limiting and quotas:** Add limits for API requests, file count, total session storage and model usage.
- **Background processing:** Move large PDF parsing/index builds to a background job queue for production workloads.
- **Observability:** Add structured logs, request IDs, latency tracking and model/retrieval diagnostics.
- **Deployment:** Configure HTTPS, secret management, persistent storage, backups and process supervision.
- **Concurrency:** Validate simultaneous uploads/deletes/chat requests and multi-worker deployment before production use.

## 15. Security and data handling notes

- Keep backend `.env` private; only `.env.example` belongs in Git.
- Do not commit uploaded documents, SQLite databases, FAISS indexes, model caches, virtual environments or `node_modules`.
- The current CORS configuration allows the local Vite development origins.
- Uploaded PDFs are stored locally in session-specific folders.
- Groq receives the question and retrieved context needed for answer generation. Do not upload confidential or regulated documents unless the relevant provider/data-handling requirements have been reviewed.
- `FAISS.load_local` uses `allow_dangerous_deserialization=True`; only load indexes created and controlled by this application. Never load an untrusted FAISS index folder.
- The prototype does not implement login or production-grade tenant authorization.

## 16. Design decisions — short summary

The implementation follows a deliberate sequence:

- **Parse before prompting:** PDFs become page-aware text documents before the LLM is involved.
- **Chunk before embedding:** Retrieval works over focused passages rather than entire documents.
- **Embed once and reuse:** The model is shared inside the API process to avoid repeated initialization.
- **Index per session:** A conversation searches only its own uploaded documents.
- **Rewrite before retrieval:** Follow-ups become standalone queries while preserving the user's intended metric and scope.
- **Retrieve before generation:** The LLM receives evidence selected from the user's PDFs.
- **Cite and abstain:** Unsupported or uncited responses are not presented as grounded answers.
- **Persist conversation state:** SQLite keeps session history and source metadata available when a conversation is reopened.

The key principle is: **retrieve evidence first, generate an answer second, and make the evidence traceable.**

---

## Author / contact

Add your name, contact email and LinkedIn/GitHub profile here before publishing.

