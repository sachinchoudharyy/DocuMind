# GitHub Repository Preparation

## 1. Before the first commit

From the DocuMind repository root, confirm:
- `.env` is not present in staged files.
- `backend/data/chat_history.db` is not staged.
- `backend/data/uploads/` and `backend/data/vectorstore/` are not staged.
- `.venv/`, `node_modules/` and build output are not staged.
- The three synthetic sample PDFs are included if you want reviewers to run the demo immediately.

The root `.gitignore` in this package is designed to exclude local secrets, runtime databases, uploads, FAISS indexes, virtual environments and frontend build output.

## 2. Add backend environment template

Copy `backend/.env.example` into the project. The real `backend/.env` remains local and must never be committed.

## 3. Initialize Git

Run from the repository root:

```powershell
git init
git branch -M main
git add .
git status
```

Inspect `git status` carefully. If `.env`, databases, uploaded private PDFs, indexes, `.venv` or `node_modules` appear, stop and fix `.gitignore` before committing.

Then:

```powershell
git commit -m "Build DocuMind multi-document RAG application"
```

## 4. Create the GitHub repository

Create an empty repository on GitHub named `DocuMind` (or another chosen name). Do not initialize it with a README if you already committed the local README.

Then connect and push:

```powershell
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/DocuMind.git
git push -u origin main
```

Replace the placeholder with your actual GitHub username and repository URL.

## 5. Suggested repository metadata

**Description**

`Multi-document PDF RAG application with session-isolated FAISS retrieval, conversational follow-up resolution, and source-cited answers.`

**Suggested topics**

`rag`, `retrieval-augmented-generation`, `llm`, `fastapi`, `react`, `typescript`, `faiss`, `langchain`, `document-ai`, `conversational-ai`

## 6. Screenshots

Add a small number of genuine screenshots under `docs/screenshots/`, for example:
- `workspace.png` — main application after upload;
- `grounded-answer.png` — answer with source/page citation;
- `abstention.png` — question not supported by uploaded documents;
- `session-isolation.png` — separate conversation with its own document set.

Only include screenshots from your actual running application. Check that they contain no API keys, personal data or private documents.

## 7. Final repository review

```powershell
git status
git ls-files
```

Review the tracked file list. The repository should contain source code, tests, documentation, safe environment templates and synthetic demo PDFs — not local runtime data or secrets.

If a secret was ever committed, deleting it in a later commit is not enough. Revoke/rotate the key and follow GitHub's sensitive-data removal guidance.
