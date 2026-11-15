from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


_embedding_model = None
_embedding_model_lock = Lock()


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the process-wide embedding model, initializing it only once.

    A lock prevents simultaneous first-time requests from loading duplicate
    copies when multiple API requests initialize indexes concurrently.
    """
    global _embedding_model

    if _embedding_model is None:
        with _embedding_model_lock:
            if _embedding_model is None:
                _embedding_model = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL,
                    model_kwargs={"device": settings.EMBEDDING_DEVICE},
                    encode_kwargs={"normalize_embeddings": True},
                )

    return _embedding_model


class EmbeddingModel:
    """Backward-compatible wrapper around the shared embedding model."""

    def __init__(self):
        self.model = get_embedding_model()

    def get_model(self):
        return self.model
