import { create } from 'zustand';
import { listDocuments, uploadDocument, deleteDocument } from '../api/client';
import type { Document } from '../types';

interface DocumentStore {
  documents: Document[];
  loading: boolean;
  highlightedIds: number[];
  fetchDocuments: () => Promise<void>;
  upload: (file: File) => Promise<void>;
  remove: (id: number) => Promise<void>;
  setHighlightedIds: (ids: number[]) => void;
}

export const useDocumentStore = create<DocumentStore>((set, get) => ({
  documents: [],
  loading: false,
  highlightedIds: [],

  fetchDocuments: async () => {
    set({ loading: true });
    try {
      const docs = await listDocuments();
      set({ documents: docs });
    } finally {
      set({ loading: false });
    }
  },

  upload: async (file: File) => {
    await uploadDocument(file);
    await get().fetchDocuments();
  },

  remove: async (id: number) => {
    await deleteDocument(id);
    set(s => ({ documents: s.documents.filter(d => d.id !== id) }));
  },

  setHighlightedIds: (ids: number[]) => set({ highlightedIds: ids }),
}));

// Expose to browser console in dev for testing without an API call:
// __setHighlightedIds([1, 2, 3])
if (import.meta.env.DEV) {
  (window as unknown as Record<string, unknown>).__setHighlightedIds =
    (ids: number[]) => useDocumentStore.getState().setHighlightedIds(ids);
}
