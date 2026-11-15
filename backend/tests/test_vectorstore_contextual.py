import re

from app.config import settings
from app.ingestion.document_loader import PDFDocumentLoader
from app.ingestion.text_splitter import DocumentChunker
from app.retrieval.retriever import ContextualRetriever
from app.vectorstore.faiss_store import FAISSVectorStore


TEST_CASES = [
    {
        "question": "What was the company's total revenue in FY 2025-26?",
        "source": "01_annual_financial_report.pdf",
        "expected": "12.5 crore",
    },
    {
        "question": "What was the net profit in FY 2024-25?",
        "source": "01_annual_financial_report.pdf",
        "expected": "1.8 crore",
    },
    {
        "question": "Which technology is used for backend APIs?",
        "source": "02_technical_architecture_report.pdf",
        "expected": "FastAPI",
    },
    {
        "question": "What is the monthly API availability target?",
        "source": "02_technical_architecture_report.pdf",
        "expected": "99.5 percent",
    },
    {
        "question": "How many paid annual leave days do employees receive?",
        "source": "03_hr_and_operational_policy.pdf",
        "expected": "18 days",
    },
]


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    print("\n" + "=" * 65)
    print("DOCUMIND - CONTEXT-AWARE RETRIEVAL EVALUATION")
    print("=" * 65)

    print("\n1. Loading PDF documents...")
    loader = PDFDocumentLoader()
    documents = loader.load_multiple(settings.SAMPLE_DOCUMENTS_DIR)

    if not documents:
        print("No documents found.")
        return

    print(f"Total extracted pages: {len(documents)}")

    print("\n2. Creating text chunks...")
    chunker = DocumentChunker()
    chunks = chunker.split_documents(documents)

    print(f"Chunk size: {settings.CHUNK_SIZE}")
    print(f"Chunk overlap: {settings.CHUNK_OVERLAP}")
    print(f"Total chunks: {len(chunks)}")

    print("\n3. Building FAISS index...")
    store = FAISSVectorStore()
    store.create_index(chunks)
    store.save_index()

    print("\n4. Creating contextual retriever...")
    retriever = ContextualRetriever(
        vector_store=store,
        all_chunks=chunks,
    )

    print("\n5. Running evaluation...")
    passed = 0
    ranks = []

    for index, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_source = test["source"]
        expected_answer = normalize_text(test["expected"])

        print("\n" + "=" * 65)
        print(f"TEST {index}: {question}")

        retrieved_docs = retriever.retrieve(
            query=question,
            k=5,
            neighbor_window=1,
        )

        expected_found = False

        for rank, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source")
            content = normalize_text(doc.page_content)
            contains_expected = expected_answer in content
            correct_source = source == expected_source
            is_expected = correct_source and contains_expected

            if is_expected:
                expected_found = True

            print(f"\nContext group rank: {rank}")
            print(f"Source: {source}")
            print(f"Page: {doc.metadata.get('page')}")
            print(f"Combined chunk IDs: {doc.metadata.get('chunk_ids')}")
            print(f"Retrieved score: {doc.metadata.get('retrieved_score')}")
            print(f"Expected answer found: {is_expected}")
            print(f"Context: {doc.page_content}")

        if expected_found:
            passed += 1
            ranks.append(1)
            print("\nRESULT: PASS - Expected answer found in contextual retrieval.")
        else:
            ranks.append(None)
            print("\nRESULT: FAIL - Expected answer not found in retrieved context.")

    total = len(TEST_CASES)
    hit_at_1 = sum(rank == 1 for rank in ranks)
    hit_at_3 = sum(rank is not None and rank <= 3 for rank in ranks)
    hit_at_5 = sum(rank is not None and rank <= 5 for rank in ranks)
    mrr = sum((1 / rank) if rank is not None else 0 for rank in ranks) / total

    print("\n" + "=" * 65)
    print("FINAL CONTEXTUAL RETRIEVAL SUMMARY")
    print("=" * 65)
    print(f"Total tests: {total}")
    print(f"Expected answers found: {passed}")
    print(f"Not found: {total - passed}")
    print(f"Context Hit@1: {hit_at_1}/{total} ({hit_at_1 / total * 100:.2f}%)")
    print(f"Context Hit@3: {hit_at_3}/{total} ({hit_at_3 / total * 100:.2f}%)")
    print(f"Context Hit@5: {hit_at_5}/{total} ({hit_at_5 / total * 100:.2f}%)")
    print(f"Context MRR: {mrr:.4f}")


if __name__ == "__main__":
    main()
