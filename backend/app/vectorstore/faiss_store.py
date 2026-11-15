from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.config import settings
from app.embeddings.embedding_model import get_embedding_model


class FAISSVectorStore:
    """FAISS wrapper supporting a separate persistent index directory per chat."""

    def __init__(self, index_path: str | Path | None = None):
        # Reuse the same process-wide model for indexing, loading, and querying.
        self.embedding_model = get_embedding_model()
        self.vector_store = None
        self.index_path = Path(index_path) if index_path else Path(settings.VECTOR_DB_DIR)

    def create_index(self, chunks):
        if not chunks:
            raise ValueError("Cannot create a FAISS index from an empty chunk list.")
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
        )
        return self.vector_store

    def save_index(self):
        if self.vector_store is None:
            raise RuntimeError("FAISS index has not been created or loaded.")
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_path))
        return self.index_path

    def load_index(self):
        if not (self.index_path / "index.faiss").exists():
            raise FileNotFoundError(f"No FAISS index found at: {self.index_path}")
        # Only load indexes created by this application in its own local data folder.
        self.vector_store = FAISS.load_local(
            str(self.index_path),
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )
        return self.vector_store

    def search(self, query: str, k: int = 5):
        if self.vector_store is None:
            raise RuntimeError("FAISS index is not loaded.")
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")
        search_query = f"{settings.QUERY_INSTRUCTION}{query.strip()}"
        return self.vector_store.similarity_search_with_score(search_query, k=k)
