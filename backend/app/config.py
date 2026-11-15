
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "DocuMind"
    APP_VERSION: str = "1.0.0"

    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-20b"

    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"

    QUERY_INSTRUCTION: str = (
        "Represent this sentence for searching "
        "relevant passages: "
    )

    CHUNK_SIZE: int = 350
    CHUNK_OVERLAP: int = 70
    RETRIEVAL_TOP_K: int = 5

    DATA_DIR: str = "data"
    SAMPLE_DOCUMENTS_DIR: str = "data/sample_documents"
    VECTOR_DB_DIR: str = "data/vectorstore"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
