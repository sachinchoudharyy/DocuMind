# DocuMind — 4 to 6 Minute Demo Script

## Before recording

1. Start the FastAPI backend and React frontend.
2. Open the application and ensure the API status is operational.
3. Have the three synthetic sample PDFs ready.
4. Close unrelated windows and hide secrets, terminal environment variables and personal files.
5. Use a clean conversation for the recording.

## Demo flow

### 1. Introduce the problem (20–30 seconds)

“DocuMind is a multi-document RAG application. It lets a user upload PDFs into a conversation and ask questions in natural language. The key idea is that answers are grounded in retrieved document passages and include source references.”

### 2. Show the workspace (20 seconds)

Point out:
- conversation sidebar;
- central chat interface;
- knowledge-base panel;
- upload area;
- source references below assistant answers.

### 3. Upload documents (30–45 seconds)

Upload:
- `01_annual_financial_report.pdf`
- `02_technical_architecture_report.pdf`
- `03_hr_and_operational_policy.pdf`

Explain that the backend validates each PDF, extracts text page by page, chunks the text, embeds the chunks and builds the current conversation's FAISS index.

### 4. Ask a factual question (30 seconds)

Ask:

“What was the company's total revenue in FY 2025-26?”

Expected sample answer: INR 12.5 crore, with the annual financial report cited.

Explain that the query is embedded, matched against the active session's FAISS index, and the retrieved passage is passed to Groq with an evidence-only answer prompt.

### 5. Demonstrate conversational memory (30 seconds)

Ask:

“And what about the previous financial year?”

Expected sample answer: INR 10 crore.

Explain that the follow-up is rewritten into a standalone question preserving the metric “revenue” while resolving the period to FY 2024-25.

### 6. Demonstrate a second domain (30 seconds)

Ask:

“How many paid annual leave days are provided to employees?”

Expected sample answer: 18 days, with the HR policy cited.

This shows retrieval across multiple uploaded PDFs.

### 7. Demonstrate abstention (30 seconds)

Ask:

“How many weeks of parental leave does the company guarantee?”

Expected behavior: DocuMind says the uploaded policy does not specify the entitlement, rather than inventing a number. The answer should be checked against the actual retrieved excerpt and citation.

Explain that empty retrieval or an answer without a valid source citation triggers a safe not-found response.

### 8. Explain isolation and deletion (45 seconds)

Create a new conversation and upload only the HR policy. Ask a finance question and show that the finance PDF from the first conversation is not available in this session.

Then delete a document and explain that the session index is rebuilt from the remaining registered PDFs.

### 9. Close with architecture (20 seconds)

“The backend uses FastAPI, PyMuPDF, LangChain, BGE embeddings, FAISS and SQLite. The frontend uses React and TypeScript. The system separates intent resolution, retrieval and answer generation so the model uses retrieved evidence rather than treating chat history as factual source material.”

## Recording checklist

- [ ] No API key or `.env` visible.
- [ ] All three sample PDFs upload successfully.
- [ ] At least one answer shows a source chip and page number.
- [ ] Follow-up question returns the previous-year metric.
- [ ] Unknown information triggers abstention.
- [ ] Session isolation is demonstrated.
- [ ] No unrelated personal information appears on screen.
