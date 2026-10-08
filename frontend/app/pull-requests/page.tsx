"use client";

import { useEffect, useState } from "react";

import { Card, GhostButton, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

type PullRequest = {
  id: string;
  number: number;
  title: string;
  state: string;
  files: { filename: string; additions: number; deletions: number }[];
};

export default function PullRequestsPage() {
  const [prs, setPrs] = useState<PullRequest[]>([]);
  const [selected, setSelected] = useState<PullRequest | null>(null);
  const [analysis, setAnalysis] = useState("");

  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((repos) => {
      if (repos[0]) {
        apiGet<PullRequest[]>(`/api/repos/${repos[0].id}/pull-requests`).then((items) => {
          setPrs(items);
          setSelected(items[0] ?? null);
        });
      }
    });
  }, []);

  async function analyze() {
    if (!selected) return;
    const data = await apiPost(`/api/pull-requests/${selected.id}/analyze`, {});
    setAnalysis(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <PageTitle title="PR Review 助手" description="读取 changed files 和 diff，生成 PR 摘要、风险点、Review Checklist 与测试建议。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-ink">Pull Requests</h2>
            <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-500">{prs.length} 个</span>
          </div>
          <div className="mt-4 space-y-2">
            {prs.map((pr) => (
              <GhostButton key={pr.id} onClick={() => setSelected(pr)} className={selected?.id === pr.id ? "border-teal-300 bg-teal-50 text-teal-900" : ""}>
                #{pr.number} {pr.title}
              </GhostButton>
            ))}
          </div>
        </Card>

        <Card>
          <h2 className="font-semibold text-ink">Review 报告</h2>
          {selected && (
            <div className="mt-3 rounded-md bg-panel p-3">
              <div className="text-sm font-medium text-ink">#{selected.number} {selected.title}</div>
              <div className="mt-3 grid gap-2">
                {selected.files.map((file) => (
                  <div key={file.filename} className="flex items-center justify-between rounded-md bg-white px-3 py-2 text-sm text-slate-600">
                    <span className="truncate">{file.filename}</span>
                    <span className="ml-3 whitespace-nowrap text-xs text-slate-500">+{file.additions} / -{file.deletions}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <PrimaryButton onClick={analyze} disabled={!selected} className="mt-4">生成 PR 分析</PrimaryButton>
          {analysis && <pre className="mt-4 max-h-[520px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{analysis}</pre>}
        </Card>
      </div>
    </>
  );
}
