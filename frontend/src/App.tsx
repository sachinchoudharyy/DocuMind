import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Activity, ArrowDown, ArrowUp, BookOpen, Check, ChevronDown, CircleHelp,
  FileText, FolderOpen, Menu, MessageSquare, MoreHorizontal, Plus, Search,
  Send, ShieldCheck, Sparkles, Trash2, UploadCloud, X, Zap,
} from "lucide-react";
import { api } from "./api";
import type { DocumentRecord, Message, Session, Source } from "./types";

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function shortDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString(undefined, {month: "short", day: "numeric"});
}
function initials(title: string) {
  return title.trim().slice(0, 1).toUpperCase() || "N";
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [docs, setDocs] = useState<DocumentRecord[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [mobileSidebar, setMobileSidebar] = useState(false);
  const [showDocs, setShowDocs] = useState(true);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const activeSession = sessions.find(s => s.session_id === activeId);

  const refreshSessions = useCallback(async () => {
    const items = await api.sessions();
    setSessions(items);
    return items;
  }, []);

  const openSession = useCallback(async (id: string) => {
    setError("");
    setActiveId(id);
    setMobileSidebar(false);
    try {
      const detail = await api.session(id);
      setMessages(detail.messages);
      setDocs(detail.documents);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to open conversation.");
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        await api.health();
        if (!mounted) return;
        setApiOnline(true);
        const items = await refreshSessions();
        if (items.length) await openSession(items[0].session_id);
        else {
          const created = await api.createSession();
          await refreshSessions();
          await openSession(created.session_id);
        }
      } catch (e) {
        if (mounted) setError(e instanceof Error ? e.message : "Unable to initialize DocuMind.");
      }
    })();
    return () => { mounted = false; };
  }, [openSession, refreshSessions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({behavior: "smooth", block: "end"});
  }, [messages, busy]);

  const filteredSessions = useMemo(() => sessions.filter(s =>
    s.title.toLowerCase().includes(search.toLowerCase())
  ), [sessions, search]);

  async function newChat() {
    try {
      setError("");
      const created = await api.createSession();
      await refreshSessions();
      await openSession(created.session_id);
    } catch (e) { setError(e instanceof Error ? e.message : "Could not create conversation."); }
  }

  async function removeChat(id: string) {
    if (!window.confirm("Delete this conversation, its messages and uploaded documents?")) return;
    try {
      await api.deleteSession(id);
      const remaining = await refreshSessions();
      if (id === activeId) {
        setMessages([]); setDocs([]);
        if (remaining.length) await openSession(remaining[0].session_id);
        else await newChat();
      }
    } catch (e) { setError(e instanceof Error ? e.message : "Could not delete conversation."); }
  }

  async function uploadFiles(fileList: FileList | File[]) {
    if (!activeId || !fileList.length) return;
    const files = Array.from(fileList);
    const invalid = files.find(f => !f.name.toLowerCase().endsWith(".pdf"));
    if (invalid) { setError(`${invalid.name} is not a PDF. Only PDF files are supported.`); return; }
    setUploading(true); setError("");
    try {
      const result = await api.upload(activeId, files);
      setDocs(result.documents);
      await refreshSessions();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload or indexing failed.");
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function removeDocument(documentId: string) {
    if (!activeId || !window.confirm("Remove this PDF and rebuild the conversation index?")) return;
    try {
      await api.deleteDocument(activeId, documentId);
      setDocs(current => current.filter(d => d.document_id !== documentId));
      await refreshSessions();
    } catch (e) { setError(e instanceof Error ? e.message : "Could not remove document."); }
  }

  async function sendMessage(text = input) {
    const question = text.trim();
    if (!question || busy || !activeId) return;
    if (!docs.length) { setError("Upload at least one PDF before asking a question."); return; }
    setError(""); setInput("");
    setMessages(current => [...current, {role: "user", content: question}]);
    setBusy(true);
    try {
      const result = await api.chat(activeId, question);
      setMessages(current => [...current, {
        role: "assistant", content: result.answer, sources: result.sources,
      }]);
      await refreshSessions();
    } catch (e) {
      setMessages(current => current.slice(0, -1));
      setInput(question);
      setError(e instanceof Error ? e.message : "The request could not be completed.");
    } finally { setBusy(false); }
  }

  const suggestions = [
    {icon: <Activity size={16}/>, title: "Executive summary", prompt: "Give me an executive summary of the uploaded documents."},
    {icon: <Search size={16}/>, title: "Find key metrics", prompt: "What are the most important figures and metrics in these documents?"},
    {icon: <BookOpen size={16}/>, title: "Compare documents", prompt: "Compare the key information across the uploaded documents."},
  ];

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileSidebar ? "sidebar-open" : ""}`}>
        <div className="brand-row">
          <div className="brand-mark"><BookOpen size={19}/></div>
          <div><div className="brand-name">documind<span>.</span></div><div className="brand-caption">DOCUMENT INTELLIGENCE</div></div>
          <button className="icon-button mobile-close" onClick={() => setMobileSidebar(false)} aria-label="Close menu"><X size={18}/></button>
        </div>
        <button className="new-chat" onClick={newChat}><Plus size={17}/> New conversation <span>⌘ K</span></button>
        <div className="sidebar-section">
          <div className="section-heading"><span>WORKSPACE</span><MoreHorizontal size={16}/></div>
          <div className="search-box"><Search size={15}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search conversations"/></div>
          <div className="conversation-list">
            {filteredSessions.map(session => (
              <div key={session.session_id} className={`conversation-item ${activeId === session.session_id ? "active" : ""}`}>
                <button className="conversation-select" onClick={() => openSession(session.session_id)}>
                  <MessageSquare size={15}/><span>{session.title}</span>
                </button>
                <button className="conversation-delete" title="Delete conversation" onClick={() => removeChat(session.session_id)}><Trash2 size={13}/></button>
              </div>
            ))}
            {!filteredSessions.length && <div className="muted-empty">No conversations found</div>}
          </div>
        </div>
        <div className="sidebar-bottom">
          <div className="plan-card">
            <div className="plan-icon"><Zap size={15}/></div>
            <div><strong>Private workspace</strong><p>Documents stay isolated by conversation.</p></div>
          </div>
          <div className="profile-row"><div className="avatar">DM</div><div className="profile-copy"><strong>DocuMind Workspace</strong><span>Local environment</span></div><MoreHorizontal size={17}/></div>
        </div>
      </aside>

      {mobileSidebar && <button className="mobile-scrim" onClick={() => setMobileSidebar(false)} aria-label="Close navigation"/>}

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" onClick={() => setMobileSidebar(true)} aria-label="Open menu"><Menu size={19}/></button>
            <div className="breadcrumb">Workspace <span>/</span> <strong>{activeSession?.title || "New conversation"}</strong></div>
          </div>
          <div className="topbar-right">
            <div className={`connection ${apiOnline ? "online" : "offline"}`}><span className="status-dot"/>{apiOnline ? "System operational" : "API disconnected"}</div>
            <div className="topbar-divider"/>
            <button className="help-button" title="DocuMind uses only the PDFs uploaded to this conversation."><CircleHelp size={17}/></button>
            <div className="avatar small">DM</div>
          </div>
        </header>

        <div className="workspace-grid">
          <section className="chat-panel">
            <div className="chat-scroll">
              {messages.length === 0 ? (
                <div className="welcome-wrap">
                  <div className="welcome-eyebrow"><span className="eyebrow-line"/> YOUR DOCUMENTS, UNDERSTOOD <span className="eyebrow-line"/></div>
                  <div className="welcome-icon"><Sparkles size={24}/></div>
                  <h1>Good documents.<br/><span>Better decisions.</span></h1>
                  <p className="welcome-description">Ask questions across your PDFs. Get clear answers backed by the exact sources and pages — not guesswork.</p>
                  <div className="welcome-features">
                    <div><ShieldCheck size={16}/> Source-grounded answers</div>
                    <div><FolderOpen size={16}/> Multi-document context</div>
                    <div><MessageSquare size={16}/> Conversational memory</div>
                  </div>
                  <div className="suggestion-grid">
                    {suggestions.map(item => <button className="suggestion-card" key={item.title} onClick={() => sendMessage(item.prompt)} disabled={!docs.length || busy}>
                      <span className="suggestion-icon">{item.icon}</span><strong>{item.title}</strong><span>{item.prompt}</span><ArrowUpRightIcon/>
                    </button>)}
                  </div>
                  {!docs.length && <div className="upload-nudge"><UploadCloud size={17}/> Upload documents in the right panel to get started.</div>}
                </div>
              ) : (
                <div className="message-list">
                  <div className="thread-date">CONVERSATION</div>
                  {messages.map((message, index) => <MessageBubble key={`${index}-${message.role}`} message={message}/>)}
                  {busy && <div className="message-row assistant-row"><div className="assistant-avatar"><Sparkles size={15}/></div><div className="message-content"><div className="thinking"><span/><span/><span/> <em>Searching your documents and composing an answer</em></div></div></div>}
                  <div ref={bottomRef}/>
                </div>
              )}
            </div>
            {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")}><X size={15}/></button></div>}
            <div className="composer-wrap">
              <div className="composer">
                <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => {if(e.key === "Enter" && !e.shiftKey){e.preventDefault();sendMessage();}}} placeholder={docs.length ? "Ask anything about your documents..." : "Upload a PDF to start asking questions..."} rows={1} disabled={!activeId || busy}/>
                <div className="composer-bottom"><div className="composer-hint"><span>↵</span> to send <span className="hint-separator">·</span> Shift + ↵ for new line</div><button className="send-button" onClick={() => sendMessage()} disabled={!input.trim() || busy || !docs.length} aria-label="Send message">{busy ? <div className="button-spinner"/> : <Send size={16}/>}</button></div>
              </div>
              <div className="disclaimer">DocuMind can make mistakes. Verify important information against the cited source documents.</div>
            </div>
          </section>

          <aside className="right-panel">
            <div className="panel-heading"><div><div className="panel-title">Knowledge base</div><div className="panel-subtitle">Sources for this conversation</div></div><button className="icon-button" onClick={() => setShowDocs(v => !v)} aria-label="Toggle documents"><ChevronDown size={17} className={showDocs ? "" : "rotated"}/></button></div>
            <div className="knowledge-stats"><div><strong>{docs.length.toString().padStart(2,"0")}</strong><span>Documents</span></div><div><strong>{docs.reduce((n,d) => n + d.size_bytes, 0) ? formatBytes(docs.reduce((n,d) => n + d.size_bytes, 0)) : "—"}</strong><span>Storage used</span></div></div>
            {showDocs && <>
              <div className={`dropzone ${dragging ? "dragging" : ""} ${uploading ? "uploading" : ""}`} onClick={() => !uploading && fileInput.current?.click()} onDragOver={e => {e.preventDefault();setDragging(true);}} onDragLeave={() => setDragging(false)} onDrop={e => {e.preventDefault();setDragging(false);uploadFiles(e.dataTransfer.files);}}>
                <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple hidden onChange={e => e.target.files && uploadFiles(e.target.files)}/>
                <div className="drop-icon">{uploading ? <div className="large-spinner"/> : <UploadCloud size={20}/>}</div>
                <strong>{uploading ? "Processing documents..." : "Add documents"}</strong>
                <span>{uploading ? "Extracting text and building the search index" : "Drag and drop PDFs, or browse files"}</span>
                <small>PDF only · Up to 20 MB per file</small>
              </div>
              <div className="documents-heading"><span>UPLOADED FILES</span><span>{docs.length} {docs.length === 1 ? "FILE" : "FILES"}</span></div>
              <div className="document-list">
                {docs.map(doc => <div className="document-item" key={doc.document_id}>
                  <div className="pdf-icon"><FileText size={17}/><span>PDF</span></div>
                  <div className="document-info"><strong title={doc.filename}>{doc.filename}</strong><span>{formatBytes(doc.size_bytes)} <i/> {shortDate(doc.uploaded_at)}</span></div>
                  <button className="document-remove" onClick={() => removeDocument(doc.document_id)} title="Remove document"><Trash2 size={14}/></button>
                  <div className="indexed-label"><Check size={11}/> Indexed</div>
                </div>)}
                {!docs.length && <div className="documents-empty"><FolderOpen size={22}/><strong>No documents yet</strong><span>Files added here become searchable in this conversation.</span></div>}
              </div>
            </>}
            <div className="panel-footer"><div className="privacy-icon"><ShieldCheck size={15}/></div><div><strong>Session-isolated retrieval</strong><p>Only files in this conversation are used to generate answers.</p></div></div>
          </aside>
        </div>
      </main>
    </div>
  );
}

function ArrowUpRightIcon() {
  return <span className="suggestion-arrow"><ArrowDown size={14}/></span>;
}

function MessageBubble({message}: {message: Message}) {
  const isUser = message.role === "user";
  return <div className={`message-row ${isUser ? "user-row" : "assistant-row"}`}>
    {!isUser && <div className="assistant-avatar"><Sparkles size={15}/></div>}
    <div className="message-content">
      <div className="message-meta">{isUser ? "YOU" : "DOCUMIND"} <span>{isUser ? "" : "· GROUNDED RESPONSE"}</span></div>
      <div className={`message-bubble ${isUser ? "user-bubble" : "assistant-bubble"}`}>
        {isUser ? <p>{message.content}</p> : <div className="markdown-content"><ReactMarkdown>{message.content}</ReactMarkdown></div>}
      </div>
      {!isUser && !!message.sources?.length && <div className="sources-block">
        <div className="sources-heading"><BookOpen size={13}/> SOURCES <span>{message.sources.length}</span></div>
        <div className="source-list">{message.sources.map((source: Source) => <div className="source-chip" key={`${source.id}-${source.source}-${source.page}`}><span className="source-id">{source.id}</span><FileText size={13}/><strong>{source.source}</strong><span>p. {source.page}</span></div>)}</div>
      </div>}
    </div>
  </div>;
}
