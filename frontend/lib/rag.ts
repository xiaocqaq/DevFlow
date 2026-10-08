export type RetrievalMethod = "vector" | "fulltext" | "hybrid";

export type KnowledgeConfig = {
  embedding_provider: string;
  embedding_model: string;
  embedding_dimensions: number;
  retrieval_method: RetrievalMethod;
  rerank_enabled: boolean;
  rerank_provider: string;
  rerank_model: string;
  top_k: number;
  score_threshold_enabled: boolean;
  score_threshold: number;
  chunk_size: number;
  chunk_overlap: number;
};

export type RuntimeModel = {
  provider: string;
  model: string;
  available: boolean;
  fallback?: string | null;
  enabled?: boolean;
  dimensions?: number;
};

export type KnowledgeBase = {
  id: string;
  name: string;
  full_name: string;
  description: string;
  provider: string;
  document_count: number;
  chunk_count: number;
  character_count: number;
  config: KnowledgeConfig;
  runtime: { embedding: RuntimeModel; rerank: RuntimeModel };
  created_at?: string;
  updated_at?: string;
};

export type KnowledgeDocument = {
  id: string;
  repo_id: string;
  name: string;
  source_type: string;
  status: "queued" | "parsing" | "splitting" | "embedding" | "completed" | "failed";
  content_type?: string;
  file_size: number;
  character_count: number;
  chunk_count: number;
  error_message?: string;
  created_at?: string;
  completed_at?: string;
};

export type KnowledgeChunk = {
  id: string;
  position: number;
  title: string;
  content: string;
  character_count: number;
  enabled: boolean;
};

export type CatalogModel = {
  provider: string;
  model: string;
  label: string;
  available: boolean;
  recommended: boolean;
  fallback?: string | null;
  dimensions?: number;
};

export type ModelCatalog = {
  embedding_models: CatalogModel[];
  rerank_models: CatalogModel[];
  deepseek: { embedding_available: boolean; rerank_available: boolean; note: string };
};

export type RetrievalResult = {
  citation: number;
  document_id: string;
  source_id: string;
  source_type: string;
  title: string;
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
  retrieval: Record<string, unknown>;
};

export type RetrievalResponse = {
  id: string;
  query: string;
  results: RetrievalResult[];
  result_count: number;
  took_ms: number;
  retrieval_config: KnowledgeConfig & {
    runtime?: unknown;
    fusion_method?: string;
    answer_gate?: {
      decision?: "answer" | "ask_clarification" | "insufficient_evidence" | "conflict";
      reason?: string;
      evidence_count?: number;
      top_score?: number;
    };
    pipeline?: {
      retrieval_method?: string;
      fusion_method?: string;
      stages?: Array<Record<string, unknown> & { name?: string }>;
    };
  };
  created_at?: string;
};

export type RetrievalHistory = {
  id: string;
  query: string;
  result_count: number;
  took_ms: number;
  created_at?: string;
};

export const statusLabel: Record<KnowledgeDocument["status"], string> = {
  queued: "排队中",
  parsing: "解析中",
  splitting: "分段中",
  embedding: "向量化",
  completed: "可用",
  failed: "失败"
};

export function formatNumber(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

export function formatBytes(value: number) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(value?: string) {
  if (!value) return "-";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(
    new Date(value)
  );
}
