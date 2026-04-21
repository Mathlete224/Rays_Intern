import { create } from 'zustand';

interface FilterStore {
  company: string;
  author: string;
  writtenDateFrom: string;
  writtenDateTo: string;

  setCompany: (v: string) => void;
  setAuthor: (v: string) => void;
  setWrittenDateFrom: (v: string) => void;
  setWrittenDateTo: (v: string) => void;
  reset: () => void;
  activeCount: () => number;
}

const defaults = {
  company: '',
  author: '',
  writtenDateFrom: '',
  writtenDateTo: '',
};

export const useFilterStore = create<FilterStore>()((set, get) => ({
  ...defaults,

  setCompany: (v) => set({ company: v }),
  setAuthor: (v) => set({ author: v }),
  setWrittenDateFrom: (v) => set({ writtenDateFrom: v }),
  setWrittenDateTo: (v) => set({ writtenDateTo: v }),
  reset: () => set(defaults),

  activeCount: () => {
    const { company, author, writtenDateFrom, writtenDateTo } = get();
    return [company, author, writtenDateFrom, writtenDateTo]
      .filter(Boolean).length;
  },
}));
