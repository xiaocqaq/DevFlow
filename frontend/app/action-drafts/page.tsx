"use client";

import { useEffect, useMemo, useState } from "react";
import { Check, CheckCircle2, Clock3, FileText, Send, ShieldAlert, X } from "lucide-react";

import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";
import { PageTitle, PrimaryButton } from "@/components/ui/card";

type ActionDraft = {
  id: string;
  repo_id: string;
  draft_type: string;
  target_type: string;
  title: string;
  content: string;
  risk_level: string;
  status: string;
  execution_result?: Record<string, unknown> | null;
  error_message?: string | null;
  created_at: string;
  executed_at?: string | null;
};

type AuditLog = { id: string; action: string; status: string; target_type?: string | null; target_id?: string | null; created_at: string };

const statusMeta: Record<string, { label: string; className: string }> = {
  pending_confirmation: { label: "等待人工确认", className: "bg-amber-50 text-amber-800 border-amber-200" },
  executed: { label: "已确认并执行", className: "bg-emerald-50 text-emerald-700 border-emerald-200" },
  cancelled: { label: "已取消", className: "bg-slate-100 text-slate-600 border-slate-200" },
  failed: { label: "执行失败", className: "bg-rose-50 text-rose-700 border-rose-200" }
};

export default function ActionDraftsPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [drafts, setDrafts] = useState<ActionDraft[]>([]);
  const [audits, setAudits] = useState<AuditLog[]>([]);
  const [title, setTitle] = useState("发布检查报告");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  async function load(repo = repoId) {
    const suffix = repo ? `?repo_id=${repo}` : "";
    const [draftItems, auditItems] = await Promise.all([
      apiGet<ActionDraft[]>(`/api/action-drafts${suffix}`),
      apiGet<AuditLog[]>(`/api/action-drafts/audit-logs${suffix}`).catch(() => [])
    ]);
    setDrafts(draftItems);
    setAudits(auditItems);
  }

  useEffect(() => {
    apiGet<Repository[]>("/api/repos")
      .then((items) => {
        setRepos(items);
        setRepoId(items[0]?.id || "");
        return load(items[0]?.id || "");
      })
      .catch(() => setRepos([]));
  }, []);

  async function createDraft() {
    if (!repoId || !content.trim()) return;
    setLoading(true);
    try {
      await apiPost<ActionDraft>("/api/action-drafts", {
        repo_id: repoId,
        draft_type: "send_report",
        target_type: "repository",
        title: title.trim() || "发布检查报告",
        content,
        risk_level: "medium",
        metadata: { source: "manual_review" }
      });
      setContent("");
      await load();
    } finally {
      setLoading(false);
    }
  }

  async function changeDraft(id: string, action: "confirm" | "cancel") {
    setLoading(true);
    try {
      await apiPost(`/api/action-drafts/${id}/${action}`, {});
      await load();
    } finally {
      setLoading(false);
    }
  }

  const pendingCount = useMemo(() => drafts.filter((draft) => draft.status === "pending_confirmation").length, [drafts]);
  const executedCount = useMemo(() => drafts.filter((draft) => draft.status === "executed").length, [drafts]);

  return (
    <>
      <PageTitle title="安全草稿与人工确认" description="Agent 的外部写操作先落成可审查草稿；只有具备批准权限的人确认后，系统才执行并写入审计记录。" />

      <section className="grid gap-3 border-y border-line bg-white px-4 py-4 sm:grid-cols-3">
        <Summary icon={FileText} label="全部草稿" value={drafts.length} />
        <Summary icon={Clock3} label="待确认" value={pendingCount} tone="amber" />
        <Summary icon={CheckCircle2} label="已执行" value={executedCount} tone="green" />
      </section>

      <div className="mt-4 grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
        <section className="self-start rounded-md border border-line bg-white p-5">
          <div className="flex items-center gap-2 text-sm font-semibold text-ink"><ShieldAlert className="h-4 w-4 text-amber-600" />创建安全草稿</div>
          <p className="mt-2 text-xs leading-5 text-slate-500">这里只保存建议内容，不会直接写入外部系统。</p>
          <label className="mt-4 block text-sm font-medium text-slate-700">仓库
            <select className="mt-2 w-full rounded-md border border-line bg-white px-3 py-2 outline-none focus:border-teal-500" value={repoId} onChange={(event) => { setRepoId(event.target.value); void load(event.target.value); }}>
              {repos.map((repo) => <option key={repo.id} value={repo.id}>{repo.full_name}</option>)}
            </select>
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">标题
            <input className="mt-2 w-full rounded-md border border-line px-3 py-2 outline-none focus:border-teal-500" value={title} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label className="mt-4 block text-sm font-medium text-slate-700">待审查内容
            <textarea className="mt-2 min-h-40 w-full resize-none rounded-md border border-line px-3 py-2 text-sm leading-6 outline-none focus:border-teal-500" value={content} onChange={(event) => setContent(event.target.value)} placeholder="输入准备发送的报告或评论内容" />
          </label>
          <PrimaryButton className="mt-4 w-full gap-2" disabled={!repoId || !content.trim() || loading} onClick={createDraft}><FileText className="h-4 w-4" />保存为待确认草稿</PrimaryButton>
        </section>

        <div className="space-y-4">
          {drafts.map((draft) => {
            const meta = statusMeta[draft.status] || { label: draft.status, className: "bg-panel text-slate-600 border-line" };
            return (
              <article key={draft.id} className="rounded-md border border-line bg-white p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="font-semibold text-ink">{draft.title || draft.draft_type}</h2>
                      <span className={`rounded-full border px-2.5 py-1 text-xs font-medium ${meta.className}`}>{meta.label}</span>
                      <span className={`rounded-full px-2.5 py-1 text-xs ${draft.risk_level === "high" ? "bg-rose-50 text-rose-700" : "bg-slate-100 text-slate-600"}`}>{draft.risk_level === "high" ? "高风险" : "中低风险"}</span>
                    </div>
                    <div className="mt-2 text-xs text-slate-500">{draft.draft_type} · {draft.target_type} · {new Date(draft.created_at).toLocaleString("zh-CN")}</div>
                  </div>
                  {draft.status === "pending_confirmation" && (
                    <div className="flex gap-2">
                      <button className="inline-flex items-center gap-1 rounded-md bg-teal-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-teal-700 disabled:opacity-60" disabled={loading} onClick={() => changeDraft(draft.id, "confirm")}><Check className="h-4 w-4" />确认执行</button>
                      <button className="inline-flex items-center gap-1 rounded-md border border-line px-3 py-2 text-sm text-slate-600 transition hover:bg-panel disabled:opacity-60" disabled={loading} onClick={() => changeDraft(draft.id, "cancel")}><X className="h-4 w-4" />取消</button>
                    </div>
                  )}
                </div>
                <div className="mt-4 whitespace-pre-wrap rounded-md border-l-4 border-teal-500 bg-panel px-4 py-3 text-sm leading-7 text-slate-700">{draft.content}</div>
                {draft.status === "executed" && <div className="mt-3 flex items-center gap-2 text-sm text-emerald-700"><CheckCircle2 className="h-4 w-4" />人工确认完成，执行结果和操作者已经写入审计记录。</div>}
                {draft.error_message && <div className="mt-3 rounded-md bg-rose-50 p-3 text-sm text-rose-700">{draft.error_message}</div>}
              </article>
            );
          })}
          {drafts.length === 0 && <div className="rounded-md border border-dashed border-line bg-white p-5 text-sm text-slate-500">暂无安全草稿。</div>}

          <section className="rounded-md border border-line bg-white p-5">
            <div className="flex items-center gap-2 font-semibold text-ink"><Send className="h-4 w-4 text-teal-700" />审计时间线</div>
            <div className="mt-4 space-y-0">
              {audits.slice(0, 10).map((item, index) => (
                <div key={item.id} className="relative flex gap-3 pb-4">
                  {index < Math.min(audits.length, 10) - 1 && <span className="absolute left-[7px] top-4 h-full w-px bg-line" />}
                  <span className={`relative mt-1 h-4 w-4 shrink-0 rounded-full border-4 border-white ${item.status === "success" ? "bg-emerald-500" : "bg-rose-500"}`} />
                  <div><div className="text-sm font-medium text-ink">{auditLabel(item.action)}</div><div className="mt-1 text-xs text-slate-500">{item.status} · {new Date(item.created_at).toLocaleString("zh-CN")}</div></div>
                </div>
              ))}
              {audits.length === 0 && <div className="text-sm text-slate-500">暂无审计记录。</div>}
            </div>
          </section>
        </div>
      </div>
    </>
  );
}

function Summary({ icon: Icon, label, value, tone = "teal" }: { icon: typeof FileText; label: string; value: number; tone?: "teal" | "amber" | "green" }) {
  const color = tone === "amber" ? "text-amber-600" : tone === "green" ? "text-emerald-600" : "text-teal-700";
  return <div className="flex items-center gap-3"><Icon className={`h-5 w-5 ${color}`} /><div><div className="text-xs text-slate-500">{label}</div><div className="mt-1 text-xl font-semibold text-ink">{value}</div></div></div>;
}

function auditLabel(action: string) {
  if (action.startsWith("action_draft.confirm")) return "人工确认并执行草稿";
  if (action === "action_draft.create") return "创建待确认草稿";
  if (action === "action_draft.cancel") return "取消草稿";
  return action;
}
