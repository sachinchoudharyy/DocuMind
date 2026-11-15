export type Session = {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  document_count: number;
};

export type DocumentRecord = {
  document_id: string;
  filename: string;
  size_bytes: number;
  uploaded_at: string;
};

export type Message = {
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  sources?: Source[];
};

export type Source = {
  id: string;
  source: string;
  page: number | string;
};

export type SessionDetail = {
  session_id: string;
  messages: Message[];
  documents: DocumentRecord[];
};
