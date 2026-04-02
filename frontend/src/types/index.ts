export interface Document {
  id: number;
  filename: string;
  total_pages: number;
  file_size_bytes: number;
  uploaded_at: string;
  processed_at: string | null;
}

export interface ChunkRef {
  chunk_id: string;
  document_id: number;
  page_number: number | null;
  metadata: Record<string, unknown>;
}

export interface AskRequest {
  question: string;
  top_k?: number;
  document_ids?: number[];
  filenames?: string[];
  page_min?: number;
  page_max?: number;
}

export interface AskResponse {
  answer: string;
  chunks_used: ChunkRef[];
}

export interface SubQueryResult {
  question: string;
  answer: string;
  chunks_used: ChunkRef[];
}

export interface AgentResult {
  goal: string;
  sub_queries: SubQueryResult[];
  synthesis: string;
}

export interface CanvasMeta {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
}

export interface CanvasDetail extends CanvasMeta {
  nodes: unknown[];
  edges: unknown[];
}

// ── Node data shapes ────────────────────────────────────────────────────────

export interface QueryNodeData {
  question: string;
  documentIds: number[];
  pageMin?: number;
  pageMax?: number;
  loading?: boolean;
}

export interface AnswerNodeData {
  question: string;
  answer: string;
  chunksUsed: ChunkRef[];
}

export interface NoteNodeData {
  text: string;
}

export interface AgentNodeData {
  goal: string;
  topK: number;
  loading?: boolean;
}

export interface SynthesisNodeData {
  goal: string;
  synthesis: string;
  subQueries: SubQueryResult[];
  expanded?: boolean;
}
