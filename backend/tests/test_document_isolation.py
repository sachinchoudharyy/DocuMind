from pathlib import Path
from tempfile import TemporaryDirectory

from app.memory.session_manager import SessionManager
from app.services.document_service import DocumentService


def main():
    project_root = Path(__file__).resolve().parents[1]
    sample_dir = project_root / "data" / "sample_documents"

    financial_pdf = sample_dir / "01_annual_financial_report.pdf"
    hr_pdf = sample_dir / "03_hr_and_operational_policy.pdf"

    if not financial_pdf.is_file() or not hr_pdf.is_file():
        raise FileNotFoundError(
            "Expected sample PDFs in data/sample_documents. "
            "Run this test from the DocuMind project root."
        )

    with TemporaryDirectory(prefix="documind_isolation_") as temp_dir:
        temp = Path(temp_dir)
        sessions = SessionManager(db_path=temp / "test.sqlite3")
        service = DocumentService(
            session_manager=sessions,
            upload_root=temp / "uploads",
            index_root=temp / "indexes",
        )

        finance_session = sessions.create_session("Finance-only")
        hr_session = sessions.create_session("HR-only")

        service.upload_pdf(
            finance_session, financial_pdf.name, financial_pdf.read_bytes()
        )
        service.upload_pdf(hr_session, hr_pdf.name, hr_pdf.read_bytes())

        finance_index, finance_chunks = service.load_session_index(finance_session)
        hr_index, hr_chunks = service.load_session_index(hr_session)

        finance_results = finance_index.search(
            "What was total revenue in FY 2025-26?", k=5
        )
        hr_results = hr_index.search(
            "How many paid annual leave days are provided?", k=5
        )

        finance_sources = {
            doc.metadata.get("source") for doc, _ in finance_results
        }
        hr_sources = {doc.metadata.get("source") for doc, _ in hr_results}

        assert finance_sources == {"01_annual_financial_report.pdf"}, finance_sources
        assert hr_sources == {"03_hr_and_operational_policy.pdf"}, hr_sources
        assert all(
            doc.metadata.get("source") != "03_hr_and_operational_policy.pdf"
            for doc in finance_chunks
        )
        assert all(
            doc.metadata.get("source") != "01_annual_financial_report.pdf"
            for doc in hr_chunks
        )

        # Verify deletion rebuilds the session index and removes its stored PDF.
        finance_docs = service.list_documents(finance_session)
        assert len(finance_docs) == 1
        assert service.delete_document(
            finance_session, finance_docs[0]["document_id"]
        )
        assert service.load_session_index(finance_session) is None

        print("PASS: each chat session indexes only its own uploaded PDFs.")
        print("PASS: document deletion removes the document from the index.")
        print("Session-specific document isolation test completed.")


if __name__ == "__main__":
    main()
