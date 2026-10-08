"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowRight, BookOpen, Database, FileText, GitBranch, Search } from "lucide-react";

import { apiGet } from "@/lib/api";
import { formatDate, formatNumber, type KnowledgeBase } from "@/lib/rag";

export default function KnowledgeBasesPage() {
  const [items, setItems] = useState<KnowledgeBase[]>([]);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    apiGet<KnowledgeBase[]>("/api/rag/repositories")
      .then(setItems)
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, []);

  const visible = items.filter((item) => `${item.name} ${item.description}`.toLowerCase().includes(query.toLowerCase()));

  return (
    <div className="mx-auto min-h-[calc(100vh-74px)] max-w-7xl px-2 py-5 sm:px-5">
      <header className="flex flex-col gap-4 border-b border-line pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="mb-1 flex items-center gap-2 text-xs font-medium text-teal-700">
            <Database className="h-4 w-4" /> RAG KNOWLEDGE
          </div>
          <h1 className="text-2xl font-semibold text-ink">项目知识</h1>
          <p className="mt-1 text-sm text-slate-500">每个 GitHub Repository 拥有独立的文档、分段和检索配置。</p>
        </div>
        <Link
          href="/"
          className="inline-flex h-10 items-center justify-center gap-2 rounded-md bg-teal-700 px-4 text-sm font-medium text-white transition hover:bg-teal-800"
        >
          <GitBranch className="h-4 w-4" /> 连接仓库
        </Link>
      </header>

      <div className="my-5 flex max-w-md items-center gap-2 rounded-md border border-line bg-white px-3 py-2.5 shadow-sm">
        <Search className="h-4 w-4 text-slate-400" />
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="搜索项目"
          className="min-w-0 flex-1 bg-transparent text-sm outline-none placeholder:text-slate-400"
        />
      </div>

      {error ? <div className="border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div> : null}
      {loading ? <div className="py-16 text-center text-sm text-slate-500">正在读取知识库...</div> : null}

      {!loading && !visible.length ? (
        <div className="flex min-h-72 flex-col items-center justify-center border border-dashed border-slate-300 bg-white px-6 text-center">
          <BookOpen className="mb-4 h-9 w-9 text-teal-700" />
          <h2 className="font-semibold text-ink">{query ? "没有匹配的项目" : "先连接一个 GitHub 仓库"}</h2>
          <p className="mt-2 max-w-md text-sm leading-6 text-slate-500">
            {query ? "换个名称或描述试试。" : "Repository 是 RAG 的项目边界。连接仓库后，再上传 Markdown、PDF、DOCX 等项目资料。"}
          </p>
          {!query ? (
            <Link href="/" className="mt-5 inline-flex items-center gap-2 text-sm font-medium text-teal-700 hover:text-teal-900">
              连接仓库 <ArrowRight className="h-4 w-4" />
            </Link>
          ) : null}
        </div>
      ) : null}

      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {visible.map((item) => (
          <Link
            key={item.id}
            href={`/knowledge/${item.id}`}
            className="group border border-line bg-white p-4 shadow-sm transition hover:border-teal-300 hover:shadow-md"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="flex min-w-0 items-center gap-3">
                <span className="grid h-10 w-10 shrink-0 place-items-center rounded-md bg-teal-50 text-teal-700">
                  <BookOpen className="h-5 w-5" />
                </span>
                <div className="min-w-0">
                  <h2 className="truncate text-sm font-semibold text-ink">{item.name}</h2>
                  <p className="mt-0.5 text-xs text-slate-400">更新于 {formatDate(item.updated_at)}</p>
                </div>
              </div>
              <ArrowRight className="mt-1 h-4 w-4 shrink-0 text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-teal-700" />
            </div>
            <p className="mt-4 line-clamp-2 min-h-10 text-sm leading-5 text-slate-600">{item.description || "暂无描述"}</p>
            <div className="mt-4 flex items-center gap-4 border-t border-slate-100 pt-3 text-xs text-slate-500">
              <span className="inline-flex items-center gap-1.5"><FileText className="h-3.5 w-3.5" /> {formatNumber(item.document_count)} 文档</span>
              <span>{formatNumber(item.chunk_count)} 分段</span>
              <span className={item.runtime.embedding.available ? "text-emerald-700" : "text-amber-700"}>
                {item.runtime.embedding.available ? "模型就绪" : "降级运行"}
              </span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
