import { useEffect, useRef, useState } from 'react';
import { askQuestion, uploadDocument } from '../api/client';
import { useChatStore } from '../store/chatStore';
import type { FilterContext } from '../store/chatStore';
import { useDocumentStore } from '../store/documentStore';
import { useFilterStore } from '../store/filterStore';
import type { ChunkRef } from '../types';

export function ChatView() {
  const { sessions, activeSessionId, activeSession, newSession, addMessage } = useChatStore();
  const { documents, fetchDocuments } = useDocumentStore();
  const {
    company, author, writtenDateFrom, writtenDateTo,
    setCompany, setAuthor, setWrittenDateFrom, setWrittenDateTo,
    reset: resetFilters, activeCount,
  } = useFilterStore();

  const session = activeSession();
  const messages = session?.messages ?? [];

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Ensure there's always an active session
  useEffect(() => {
    if (!activeSessionId && sessions.length === 0) {
      newSession('Chat 1');
    } else if (!activeSessionId && sessions.length > 0) {
      useChatStore.getState().setActiveSession(sessions[0].id);
    }
  }, [activeSessionId, sessions, newSession]);

  function getOrCreateSessionId(): string {
    if (activeSessionId) return activeSessionId;
    return newSession();
  }

  async function handleSend() {
    const question = input.trim();
    if (!question || loading) return;

    const sessionId = getOrCreateSessionId();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    addMessage(sessionId, { id: `${Date.now()}`, role: 'user', content: question });
    setLoading(true);
    const filterContext = buildFilterContext();

    try {
      const result = await askQuestion({
        question,
        sender_companies: company ? [company] : undefined,
        sender_names: author ? [author] : undefined,
        written_date_from: writtenDateFrom || undefined,
        written_date_to: writtenDateTo || undefined,
      });
      addMessage(sessionId, {
        id: `${Date.now() + 1}`,
        role: 'assistant',
        content: result.answer,
        chunks: result.chunks_used,
        filterContext,
      });
    } catch {
      addMessage(sessionId, {
        id: `${Date.now() + 1}`,
        role: 'system',
        content: 'Error getting answer. Please try again.',
      });
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length === 0) return;

    const sessionId = getOrCreateSessionId();
    setUploading(true);

    addMessage(sessionId, {
      id: `${Date.now()}`,
      role: 'system',
      content: `Uploading ${files.length} file${files.length > 1 ? 's' : ''}... This may take several minutes.`,
    });

    for (const file of files) {
      try {
        const result = await uploadDocument(file);
        addMessage(sessionId, {
          id: `${Date.now()}-${file.name}`,
          role: 'system',
          content: result.status === 'skipped'
            ? `⚠️ "${file.name}" is already in the database — skipped.`
            : `✓ "${file.name}" ingested and ready to query.`,
        });
      } catch {
        addMessage(sessionId, {
          id: `${Date.now()}-${file.name}-err`,
          role: 'system',
          content: `✗ "${file.name}" failed to upload. Please try again.`,
        });
      }
    }

    setUploading(false);
    if (fileRef.current) fileRef.current.value = '';
    await fetchDocuments();
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleTextareaChange(e: React.ChangeEvent<HTMLTextAreaElement>) {
    setInput(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = `${Math.min(e.target.scrollHeight, 128)}px`;
  }

  const filterCount = activeCount();
  const docMap = Object.fromEntries(documents.map(d => [d.id, d.filename]));

  function buildFilterContext(): FilterContext | undefined {
    const ctx: FilterContext = {};
    if (company) ctx.senderCompanies = [company];
    if (author) ctx.senderNames = [author];
    if (writtenDateFrom) ctx.writtenDateFrom = writtenDateFrom;
    if (writtenDateTo) ctx.writtenDateTo = writtenDateTo;
    return Object.keys(ctx).length > 0 ? ctx : undefined;
  }

  function formatFilterContext(ctx: FilterContext): string {
    const parts: string[] = [];
    if (ctx.senderCompanies?.length) parts.push(ctx.senderCompanies.join(', '));
    if (ctx.senderNames?.length) parts.push(`by ${ctx.senderNames.join(', ')}`);
    if (ctx.writtenDateFrom || ctx.writtenDateTo)
      parts.push(`written ${ctx.writtenDateFrom ?? '…'}–${ctx.writtenDateTo ?? '…'}`);
    return parts.join(' · ');
  }

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <p className="text-lg font-medium mb-1">Ask a question</p>
              <p className="text-sm">Upload a PDF using the paperclip, then start asking questions</p>
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            {msg.role === 'system' ? (
              <div className="text-xs text-gray-400 bg-gray-50 border border-gray-100 rounded-lg px-3 py-2 max-w-xl mx-auto text-center">
                {msg.content}
              </div>
            ) : (
              <div className={`flex flex-col gap-1 max-w-2xl ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <div className={`rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap ${
                  msg.role === 'user'
                    ? 'bg-blue-500 text-white rounded-br-sm'
                    : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                }`}>
                  {msg.content}
                </div>
                {msg.chunks && msg.chunks.length > 0 && (
                  <div className="flex flex-wrap gap-1 px-1">
                    {msg.chunks.map((c: ChunkRef, i: number) => (
                      <span key={i} className="text-xs bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">
                        {docMap[c.document_id] ?? `doc ${c.document_id}`}
                        {c.page_number != null ? ` p.${c.page_number}` : ''}
                      </span>
                    ))}
                  </div>
                )}
                {msg.role === 'assistant' && msg.filterContext && (
                  <p className="text-xs text-gray-400 px-1 flex items-center gap-1">
                    <svg className="w-3 h-3 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                        d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2a1 1 0 01-.293.707L13 13.414V19a1 1 0 01-.553.894l-4 2A1 1 0 017 21v-7.586L3.293 6.707A1 1 0 013 6V4z" />
                    </svg>
                    {formatFilterContext(msg.filterContext)}
                  </p>
                )}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-2xl rounded-bl-sm px-4 py-3">
              <div className="flex gap-1 items-center">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Active filter chips — sourced from global filterStore */}
      {filterCount > 0 && (
        <div className="border-t border-gray-100 px-4 py-2 flex flex-wrap gap-1.5 items-center bg-blue-50">
          <span className="text-xs text-blue-400 font-medium shrink-0">Filtering:</span>
          {company && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-blue-700 border border-blue-200 px-2 py-0.5 rounded-full">
              {company}
              <button onClick={() => setCompany('')} className="text-blue-400 hover:text-blue-600 leading-none">×</button>
            </span>
          )}
          {author && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-purple-700 border border-purple-200 px-2 py-0.5 rounded-full">
              by {author}
              <button onClick={() => setAuthor('')} className="text-purple-400 hover:text-purple-600 leading-none">×</button>
            </span>
          )}
          {(writtenDateFrom || writtenDateTo) && (
            <span className="inline-flex items-center gap-1 text-xs bg-white text-gray-700 border border-gray-200 px-2 py-0.5 rounded-full">
              written {writtenDateFrom || '…'}–{writtenDateTo || '…'}
              <button onClick={() => { setWrittenDateFrom(''); setWrittenDateTo(''); }} className="text-gray-400 hover:text-gray-600 leading-none">×</button>
            </span>
          )}
          <button onClick={resetFilters} className="text-xs text-blue-400 hover:text-blue-600 ml-1">
            Clear all
          </button>
        </div>
      )}

      {/* Input bar */}
      <div className="border-t border-gray-200 p-4">
        <div className="flex items-end gap-2 bg-gray-50 border border-gray-200 rounded-2xl px-3 py-2">
          {/* Upload button */}
          <button
            onClick={() => fileRef.current?.click()}
            disabled={uploading}
            title="Upload PDF(s)"
            className="text-gray-400 hover:text-blue-500 disabled:opacity-40 transition-colors p-1 shrink-0 mb-0.5"
          >
            {uploading ? (
              <div className="w-5 h-5 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
              </svg>
            )}
          </button>
          <input ref={fileRef} type="file" accept=".pdf" multiple className="hidden" onChange={handleUpload} />

          {/* Text input */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 bg-transparent resize-none text-sm text-gray-800 placeholder-gray-400 outline-none"
            style={{ maxHeight: '128px' }}
          />

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={!input.trim() || loading}
            className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-200 text-white disabled:text-gray-400 rounded-xl p-1.5 shrink-0 transition-colors mb-0.5"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-1.5 px-1">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}
