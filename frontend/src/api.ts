import type { DocumentRecord, Session, SessionDetail, Source } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error("Cannot connect to DocuMind API. Check that the backend is running.");
  }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch { /* keep fallback */ }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{status: string}>("/health"),
  sessions: async () => (await request<{sessions: Session[]}>("/sessions")).sessions,
  createSession: (title = "New conversation") =>
    request<{session_id: string; title: string}>("/sessions", {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({title}),
    }),
  session: (id: string) => request<SessionDetail>(`/sessions/${id}`),
  deleteSession: (id: string) => request<void>(`/sessions/${id}`, {method: "DELETE"}),
  upload: async (id: string, files: File[]) => {
    const data = new FormData();
    files.forEach(file => data.append("files", file));
    return request<{uploaded: {document_id: string; filename: string}[]; documents: DocumentRecord[]}>(
      `/sessions/${id}/documents`, {method: "POST", body: data},
    );
  },
  deleteDocument: (sessionId: string, documentId: string) =>
    request<void>(`/sessions/${sessionId}/documents/${documentId}`, {method: "DELETE"}),
  chat: (id: string, message: string) =>
    request<{answer: string; sources: Source[]; resolved_question: string}>(
      `/sessions/${id}/chat`, {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({message}),
      },
    ),
};
