# DocuMind Architecture

## System architecture

```mermaid
flowchart TB
    U[User] --> UI[React + TypeScript UI<br/>Vite]
    UI -->|REST / JSON / multipart PDF| API[FastAPI API]

    subgraph Backend[Python backend]
      API --> SVC[Document Service]
      API --> SM[Session Manager]
      SM --> DB[(SQLite<br/>sessions · messages · documents)]

      SVC --> VAL[PDF validation]
      VAL --> LOAD[PyMuPDF<br/>page-level text extraction]
      LOAD --> CHUNK[Recursive Character Splitter<br/>350 chars · 70 overlap]
      CHUNK --> EMB[Shared BGE Embedding Model<br/>CPU · normalized vectors]
      EMB --> IDX[(FAISS index<br/>one index per session)]

      API --> HIST[Recent chat history<br/>up to 10 messages]
      HIST --> REWRITE[Follow-up question rewrite<br/>Groq LLM]
      REWRITE --> QEMB[Query embedding]
      QEMB --> IDX
      IDX --> RET[Contextual Retriever<br/>top-k + same-page neighbors]
      RET --> CTX[Bounded evidence context<br/>source/page labels]
      CTX --> GEN[Groq answer generation<br/>evidence-only prompt]
      GEN --> GUARD[Citation normalization<br/>valid-ID check · abstention]
      GUARD --> API
    end

    API -->|Answer · cited sources · resolved question| UI
    UI --> R[Rendered response + source chips]
```

## Upload / indexing sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as React UI
    participant API as FastAPI
    participant DS as DocumentService
    participant PDF as PyMuPDF + Chunker
    participant EMB as Shared BGE Model
    participant FAISS
    participant DB as SQLite

    User->>UI: Select PDF(s)
    UI->>API: POST /sessions/{id}/documents
    API->>DS: Validate and store PDF
    DS->>DB: Register document metadata
    DS->>PDF: Extract pages and create chunks
    PDF->>EMB: Embed chunks
    EMB->>FAISS: Create session-specific index
    FAISS-->>DS: Save index
    DS-->>API: Uploaded + indexed result
    API-->>UI: Document list and status
```

## Question / answer sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as React UI
    participant API as FastAPI
    participant DB as SQLite
    participant RAG as DocumentRAGChain
    participant LLM as Groq
    participant RET as ContextualRetriever
    participant VDB as Session FAISS

    User->>UI: Ask question
    UI->>API: POST /sessions/{id}/chat
    API->>DB: Load recent messages
    API->>RAG: question + history
    opt Follow-up / history exists
      RAG->>LLM: Rewrite as standalone query
      LLM-->>RAG: Resolved question
    end
    RAG->>RET: Retrieve resolved question
    RET->>VDB: Embed query + similarity search
    VDB-->>RET: Top-k matching chunks
    RET-->>RAG: Matching chunks + nearby same-page context
    RAG->>LLM: Resolved question + labelled evidence
    LLM-->>RAG: Draft answer with source labels
    RAG->>RAG: Normalize and validate citation IDs
    alt No retrieval or no valid citation
      RAG-->>API: Abstention + no sources
    else Valid citation present
      RAG-->>API: Answer + cited sources
    end
    API->>DB: Persist user and assistant messages
    API-->>UI: Answer + sources + resolved question
    UI-->>User: Render answer and source chips
```

## Trust boundaries and data ownership

- A conversation owns its document records, uploaded files and FAISS index.
- The SQLite database stores session metadata, messages, citation metadata and document records.
- The embedding model runs in the backend process and is reused within that process.
- The Groq API is an external model service. The application sends the resolved question and retrieved context for generation.
- The UI does not query FAISS or SQLite directly; it uses the FastAPI endpoints.

## Runtime storage layout

```text
backend/data/
├── chat_history.db
├── uploads/
│   └── <session_id>/
│       └── <generated_id>_<original_filename>.pdf
└── vectorstore/
    └── <session_id>/
        ├── index.faiss
        └── index.pkl
```

Actual runtime contents are generated locally and should not be committed to Git.
