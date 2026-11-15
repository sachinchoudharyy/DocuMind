from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.chains.rag_chain import DocumentRAGChain
from app.config import settings
from app.memory.session_manager import SessionManager
from app.services.document_service import DocumentService


class CreateSessionRequest(BaseModel):
    title: str = Field(default="New conversation", max_length=120)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    resolved_question: str


sessions = SessionManager()
documents = DocumentService(session_manager=sessions)
_chain_cache: dict[str, DocumentRAGChain] = {}
_cache_lock = RLock()


def get_chain(session_id: str) -> DocumentRAGChain:
    with _cache_lock:
        chain = _chain_cache.get(session_id)
        if chain is not None:
            return chain

        loaded = documents.load_session_index(session_id)
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Upload at least one PDF before asking a question.",
            )
        vector_store, chunks = loaded
        chain = DocumentRAGChain(vector_store=vector_store, all_chunks=chunks)
        _chain_cache[session_id] = chain
        return chain


def invalidate_chain(session_id: str) -> None:
    with _cache_lock:
        _chain_cache.pop(session_id, None)


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    yield
    with _cache_lock:
        _chain_cache.clear()


app = FastAPI(
    title="DocuMind API",
    version=settings.APP_VERSION,
    description="Session-isolated, document-grounded conversational retrieval API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": settings.APP_NAME, "version": settings.APP_VERSION}


@app.get("/api/sessions")
def list_sessions():
    return {"sessions": sessions.list_sessions()}


@app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
def create_session(payload: CreateSessionRequest):
    session_id = sessions.create_session(payload.title)
    return {"session_id": session_id, "title": payload.title.strip() or "New conversation"}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if not sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {
        "session_id": session_id,
        "messages": sessions.get_messages(session_id),
        "documents": documents.list_documents(session_id),
    }


@app.delete("/api/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(session_id: str):
    if not sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    invalidate_chain(session_id)
    documents.delete_session_files(session_id)
    sessions.delete_session(session_id)
    return None


@app.get("/api/sessions/{session_id}/documents")
def list_documents(session_id: str):
    if not sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"documents": documents.list_documents(session_id)}


@app.post("/api/sessions/{session_id}/documents", status_code=status.HTTP_201_CREATED)
async def upload_documents(
    session_id: str,
    files: Annotated[list[UploadFile], File(...)],
):
    if not sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    if not files:
        raise HTTPException(status_code=400, detail="Select at least one PDF.")

    uploaded = []
    try:
        for file in files:
            content = await file.read()
            result = documents.upload_pdf(
                session_id=session_id,
                filename=file.filename or "document.pdf",
                content=content,
            )
            uploaded.append(result)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Document processing failed.") from exc
    finally:
        for file in files:
            await file.close()

    invalidate_chain(session_id)
    return {"uploaded": uploaded, "documents": documents.list_documents(session_id)}


@app.delete(
    "/api/sessions/{session_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_document(session_id: str, document_id: str):
    if not sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        deleted = documents.delete_document(session_id, document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not rebuild the document index.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    invalidate_chain(session_id)
    return None


@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
def chat(session_id: str, payload: ChatRequest):
    if not sessions.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Conversation not found.")
    question = payload.message.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    history = sessions.get_history_text(session_id, max_messages=10)
    try:
        chain = get_chain(session_id)
        result = chain.ask(question=question, chat_history=history)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="The AI service could not complete this request.") from exc

    sessions.add_message(session_id, "user", question)
    sessions.add_message(session_id, "assistant", result["answer"], sources=result.get("sources", []))
    return result
