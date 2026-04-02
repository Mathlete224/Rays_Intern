import { create } from 'zustand';
import { listDocuments, uploadDocument, deleteDocument } from '../api/client';
import type { Document } from '../types';

interface DocumentStore {
  documents: Document[];
  loading: boolean;
  fetchDocuments: () => Promise<void>;
  upload: (file: File) => Promise<void>;
  remove: (id: number) => Promise<void>;
}

export const useDocumentStore = create<DocumentStore>((set, get) => ({
  documents: [],
  loading: false,

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
}));
