
from pathlib import Path
import re
import pymupdf

from langchain_core.documents import Document


class PDFDocumentLoader:

    @staticmethod
    def clean_text(text: str) -> str:

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            # Remove repeated synthetic PDF footer
            if "Synthetic DocuMind evaluation data" in line:
                continue

            # Remove standalone page numbers
            if re.fullmatch(r"Page\s+\d+", line, re.IGNORECASE):
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()

    def load_pdf(self, file_path: str) -> list[Document]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported"
            )

        documents = []

        try:

            with pymupdf.open(path) as pdf:

                for page_number, page in enumerate(
                    pdf, start=1
                ):

                    raw_text = page.get_text("text")

                    cleaned_text = self.clean_text(
                        raw_text
                    )

                    if not cleaned_text:
                        continue

                    documents.append(
                        Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": path.name,
                                "page": page_number,
                                "file_path": str(
                                    path.resolve()
                                )
                            }
                        )
                    )

        except Exception as exc:

            raise RuntimeError(
                f"Failed to process {path.name}: {exc}"
            ) from exc

        return documents

    def load_multiple(
        self,
        directory: str
    ) -> list[Document]:

        folder = Path(directory)

        if not folder.exists():
            raise FileNotFoundError(
                f"Directory not found: {folder}"
            )

        pdf_files = sorted(folder.glob("*.pdf"))

        all_documents = []

        for pdf_path in pdf_files:

            try:

                documents = self.load_pdf(
                    str(pdf_path)
                )

                all_documents.extend(documents)

                print(
                    f"Loaded {pdf_path.name}: "
                    f"{len(documents)} pages"
                )

            except Exception as exc:

                print(
                    f"Skipping {pdf_path.name}: {exc}"
                )

        return all_documents
