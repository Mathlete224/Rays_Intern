import { useEffect, useRef, useState } from 'react';
import { useDocumentStore } from '../store/documentStore';
import { useCanvasStore } from '../store/canvasStore';

export function Sidebar() {
  const { documents, loading, fetchDocuments, upload, remove } = useDocumentStore();
  const { savedCanvases, fetchSavedCanvases, loadCanvas, removeCanvas, newCanvas } = useCanvasStore();
  const fileRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  useEffect(() => {
    fetchDocuments();
    fetchSavedCanvases();
  }, [fetchDocuments, fetchSavedCanvases]);

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadError(null);
    try {
      await upload(file);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Upload failed';
      setUploadError(msg);
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = '';
    }
  }

  return (
    <div className="w-64 bg-gray-50 border-r border-gray-200 flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto">

        {/* Documents section */}
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Documents</h2>
            <button
              onClick={() => { setUploadError(null); fileRef.current?.click(); }}
              disabled={uploading}
              className="text-xs bg-blue-500 hover:bg-blue-600 disabled:bg-blue-300 text-white px-2 py-1 rounded-md transition-colors"
            >
              + Upload
            </button>
            <input ref={fileRef} type="file" accept=".pdf" className="hidden" onChange={handleUpload} />
          </div>

          {/* Upload progress banner */}
          {uploading && (
            <div className="mb-2 bg-blue-50 border border-blue-200 rounded-lg p-2">
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin shrink-0" />
                <span className="text-xs font-medium text-blue-700">Processing PDF…</span>
              </div>
              <p className="text-xs text-blue-500 leading-tight">
                Docling is parsing the document and Gemini is verbalizing charts. This takes <strong>1–3 minutes</strong> — please wait.
              </p>
            </div>
          )}

          {uploadError && (
            <div className="mb-2 bg-red-50 border border-red-200 rounded-lg p-2 text-xs text-red-600">
              {uploadError}
            </div>
          )}

          {loading ? (
            <p className="text-xs text-gray-400 text-center py-2">Loading…</p>
          ) : documents.length === 0 && !uploading ? (
            <p className="text-xs text-gray-400 text-center py-2">No documents yet</p>
          ) : (
            <ul className="space-y-1">
              {documents.map(doc => (
                <li
                  key={doc.id}
                  className="flex items-start justify-between gap-1 bg-white rounded-lg px-2 py-1.5 text-xs border border-gray-100"
                >
                  <div className="min-w-0">
                    <p className="font-medium text-gray-700 truncate" title={doc.filename}>
                      {doc.filename}
                    </p>
                    <p className="text-gray-400">{doc.total_pages} pages</p>
                  </div>
                  <button
                    onClick={() => remove(doc.id)}
                    className="text-gray-300 hover:text-red-400 shrink-0 mt-0.5"
                    title="Delete"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Canvases section */}
        <div className="p-3 border-b border-gray-200">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">Canvases</h2>
            <button
              onClick={() => {
                const name = prompt('Canvas name:', 'Untitled Canvas');
                if (name) newCanvas(name);
              }}
              className="text-xs bg-purple-500 hover:bg-purple-600 text-white px-2 py-1 rounded-md transition-colors"
            >
              + New
            </button>
          </div>

          {savedCanvases.length === 0 ? (
            <p className="text-xs text-gray-400 text-center py-2">No saved canvases</p>
          ) : (
            <ul className="space-y-1">
              {savedCanvases.map(c => (
                <li
                  key={c.id}
                  className="flex items-center justify-between gap-1 bg-white rounded-lg px-2 py-1.5 text-xs border border-gray-100"
                >
                  <button
                    className="text-gray-700 hover:text-purple-600 font-medium truncate text-left flex-1"
                    onClick={() => loadCanvas(c.id)}
                    title={c.name}
                  >
                    {c.name}
                  </button>
                  <button
                    onClick={() => {
                      if (confirm(`Delete canvas "${c.name}"?`)) removeCanvas(c.id);
                    }}
                    className="text-gray-300 hover:text-red-400 shrink-0"
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* How to use */}
        <div className="p-3">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">How to use</h2>
          <ul className="space-y-2 text-xs text-gray-500">
            <li className="flex gap-2">
              <span className="shrink-0">1.</span>
              <span>Upload a PDF above</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">2.</span>
              <span><strong className="text-gray-600">Right-click</strong> the canvas to add a node</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">3.</span>
              <span><strong className="text-gray-600">Query node</strong> — ask a question, get an answer</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">4.</span>
              <span><strong className="text-gray-600">Agent node</strong> — give a high-level goal, auto-generates multi-step research</span>
            </li>
            <li className="flex gap-2">
              <span className="shrink-0">5.</span>
              <span>Drag edges between nodes to link findings</span>
            </li>
          </ul>
        </div>

      </div>
    </div>
  );
}
