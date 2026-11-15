from app.config import settings
from app.ingestion.document_loader import PDFDocumentLoader
from app.ingestion.text_splitter import DocumentChunker
from app.vectorstore.faiss_store import FAISSVectorStore
from app.chains.rag_chain import DocumentRAGChain


def main():
    print("\n" + "=" * 65)
    print("DOCUMIND - END-TO-END RAG TEST")
    print("=" * 65)

    print("\n1. Loading sample PDFs...")
    loader = PDFDocumentLoader()
    documents = loader.load_multiple(settings.SAMPLE_DOCUMENTS_DIR)

    if not documents:
        print("No PDFs found in:", settings.SAMPLE_DOCUMENTS_DIR)
        return

    print(f"Loaded {len(documents)} PDF pages.")

    print("\n2. Splitting documents...")
    chunker = DocumentChunker()
    chunks = chunker.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("\n3. Building FAISS index...")
    store = FAISSVectorStore()
    store.create_index(chunks)
    store.save_index()

    print("\n4. Initializing RAG chain...")
    rag = DocumentRAGChain(
        vector_store=store,
        all_chunks=chunks,
        top_k=settings.RETRIEVAL_TOP_K,
        neighbor_window=1,
    )

    questions = [
        "What was the company's total revenue in FY 2025-26?",
        "What was the net profit in FY 2024-25?",
        "How many paid annual leave days do employees receive?",
        "Does the policy specify parental leave entitlement?",
    ]

    chat_history = ""

    print("\n5. Asking questions...")
    for question in questions:
        print("\n" + "=" * 65)
        print("QUESTION:", question)

        result = rag.ask(
            question=question,
            chat_history=chat_history,
        )

        print("\nANSWER:")
        print(result["answer"])

        print("\nSOURCE REFERENCES:")
        if result["sources"]:
            for source in result["sources"]:
                print(
                    f"[{source['id']}] "
                    f"{source['source']} - Page {source['page']}"
                )
        else:
            print("No sources retrieved.")

        chat_history += (
            f"\nUser: {question}\nAssistant: {result['answer']}\n"
        )


if __name__ == "__main__":
    main()
