
from app.ingestion.document_loader import PDFDocumentLoader
from app.ingestion.text_splitter import DocumentChunker


def main():

    loader = PDFDocumentLoader()

    documents = loader.load_multiple(
        "data/sample_documents"
    )

    if not documents:
        print("No PDF documents found.")
        return

    print("\nTotal extracted pages:", len(documents))

    chunker = DocumentChunker()

    chunks = chunker.split_documents(documents)

    print("Total chunks:", len(chunks))

    for chunk in chunks[:3]:

        print("\nSource:", chunk.metadata["source"])
        print("Page:", chunk.metadata["page"])
        print("Chunk ID:", chunk.metadata["chunk_id"])
        print("Content:", chunk.page_content[:200])


if __name__ == "__main__":
    main()
