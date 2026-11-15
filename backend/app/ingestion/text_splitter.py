
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import Document

from app.config import settings


class DocumentChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ],
            length_function=len,
            add_start_index=True
        )

    def split_documents(
        self,
        documents: list[Document]
    ) -> list[Document]:

        if not documents:
            return []

        chunks = self.splitter.split_documents(
            documents
        )

        for index, chunk in enumerate(chunks):

            chunk.metadata["chunk_id"] = index

        return chunks
