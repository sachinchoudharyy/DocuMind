from app.chains.rag_chain import DocumentRAGChain
from app.config import settings
from app.ingestion.document_loader import PDFDocumentLoader
from app.ingestion.text_splitter import DocumentChunker
from app.vectorstore.faiss_store import FAISSVectorStore
from app.memory.session_manager import SessionManager


def main():
    print("\n" + "=" * 70)
    print("DOCUMIND - CONVERSATIONAL MEMORY TEST")
    print("=" * 70)

    print("\n1. Loading sample PDFs...")
    loader = PDFDocumentLoader()
    documents = loader.load_multiple(settings.SAMPLE_DOCUMENTS_DIR)

    if not documents:
        print("No PDFs found. Add PDFs to:", settings.SAMPLE_DOCUMENTS_DIR)
        return

    print(f"Loaded {len(documents)} pages.")

    print("\n2. Creating chunks and FAISS index...")
    chunks = DocumentChunker().split_documents(documents)
    store = FAISSVectorStore()
    store.create_index(chunks)
    store.save_index()
    print(f"Indexed {len(chunks)} chunks.")

    print("\n3. Initializing RAG and SQLite session storage...")
    rag = DocumentRAGChain(
        vector_store=store,
        all_chunks=chunks,
        top_k=settings.RETRIEVAL_TOP_K,
        neighbor_window=1,
    )

    sessions = SessionManager()
    session_id = sessions.create_session("DocuMind memory assessment")
    print("Created session:", session_id)
    print("SQLite database:", sessions.db_path)

    # Turn 1: ask a question and persist both sides of the exchange.
    question_1 = "What was the company's total revenue in FY 2025-26?"
    print("\n" + "=" * 70)
    print("TURN 1")
    print("USER:", question_1)

    history_1 = sessions.get_history_text(session_id)
    sessions.add_message(session_id, "user", question_1)

    result_1 = rag.ask(
        question=question_1,
        chat_history=history_1,
    )

    sessions.add_message(
        session_id,
        "assistant",
        result_1["answer"],
    )

    print("DOCUMIND:", result_1["answer"])
    print("RESOLVED QUESTION:", result_1["resolved_question"])

    # Turn 2: deliberately ambiguous follow-up.
    question_2 = "And what about the previous financial year?"
    print("\n" + "=" * 70)
    print("TURN 2 - FOLLOW-UP")
    print("USER:", question_2)

    history_2 = sessions.get_history_text(session_id)
    sessions.add_message(session_id, "user", question_2)

    result_2 = rag.ask(
        question=question_2,
        chat_history=history_2,
    )

    sessions.add_message(
        session_id,
        "assistant",
        result_2["answer"],
    )

    print("RESOLVED QUESTION:", result_2["resolved_question"])
    print("DOCUMIND:", result_2["answer"])

    print("\nSOURCES:")
    for source in result_2["sources"]:
        print(
            f"[{source['id']}] "
            f"{source['source']} - Page {source['page']}"
        )

    # Verify session isolation: a new session must not inherit old messages.
    print("\n" + "=" * 70)
    print("TURN 3 - SESSION ISOLATION CHECK")
    second_session_id = sessions.create_session("Isolation test")
    second_session_history = sessions.get_history_text(second_session_id)

    print("First session message count:", len(sessions.get_messages(session_id)))
    print("Second session history:", second_session_history)

    if second_session_history == "No previous conversation.":
        print("PASS: New session does not see another session's messages.")
    else:
        print("FAIL: Session history was not isolated.")

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)

    for session in sessions.list_sessions()[:5]:
        print(
            f"Session: {session['session_id']} | "
            f"Title: {session['title']} | "
            f"Messages: {session['message_count']}"
        )

    print("\nChat history persisted at:", sessions.db_path)
    print("Test completed.")


if __name__ == "__main__":
    main()
