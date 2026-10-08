"use client";

import { useEffect, useState } from "react";

import { Card, GhostButton, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPost } from "@/lib/api";
import type { Repository } from "@/lib/types";

type WorkflowRun = {
  id: string;
  name: string;
  status: string;
  conclusion?: string | null;
  logs_text?: string | null;
  jobs?: Array<{ name?: string; conclusion?: string; steps?: Array<{ name?: string; conclusion?: string }> }>;
};

export default function CIPage() {
  const [runs, setRuns] = useState<WorkflowRun[]>([]);
  const [selected, setSelected] = useState<WorkflowRun | null>(null);
  const [analysis, setAnalysis] = useState("");

  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((repos) => {
      if (repos[0]) {
        apiGet<WorkflowRun[]>(`/api/repos/${repos[0].id}/workflow-runs`).then((items) => {
          setRuns(items);
          setSelected(items.find((item) => item.conclusion === "failure") ?? items[0] ?? null);
        });
      }
    });
  }, []);

  async function analyze() {
    if (!selected) return;
    const data = await apiPost(`/api/workflow-runs/${selected.id}/analyze`, {});
    setAnalysis(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <PageTitle title="CI Debug 分析" description="聚合 GitHub Actions 运行状态和失败日志，生成失败归因、排查步骤和相关风险提示。" />
      <div className="grid gap-4 lg:grid-cols-[380px_1fr]">
        <Card>
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-ink">Workflow Runs</h2>
            <span className="rounded-full bg-panel px-2.5 py-1 text-xs text-slate-500">{runs.length} 次</span>
          </div>
          <div className="mt-4 space-y-2">
            {runs.map((run) => (
              <GhostButton key={run.id} onClick={() => setSelected(run)} className={selected?.id === run.id ? "border-teal-300 bg-teal-50 text-teal-900" : ""}>
                <div className="flex items-center justify-between gap-3">
                  <span>{run.name}</span>
                  <span className={run.conclusion === "failure" ? "text-amber-700" : "text-emerald-700"}>{run.conclusion ?? run.status}</span>
                </div>
              </GhostButton>
            ))}
          </div>
        </Card>

        <Card>
          <h2 className="font-semibold text-ink">排查建议</h2>
          {(selected?.jobs?.length ?? 0) > 0 && (
            <div className="mt-3 rounded-md border border-line bg-panel p-3 text-xs leading-5 text-slate-700">
              <div className="font-medium text-ink">失败 Job</div>
              {(selected?.jobs ?? [])
                .filter((job) => job.conclusion === "failure")
                .map((job) => (
                  <div key={job.name} className="mt-2">
                    <span className="font-medium">{job.name}</span>
                    {job.steps?.filter((step) => step.conclusion === "failure").map((step) => (
                      <span key={step.name} className="ml-2 rounded-full bg-amber-100 px-2 py-0.5 text-amber-800">
                        {step.name}
                      </span>
                    ))}
                  </div>
                ))}
            </div>
          )}
          {selected?.logs_text && <pre className="mt-3 max-h-44 overflow-auto rounded-md border border-line bg-panel p-3 text-xs text-slate-700">{selected.logs_text}</pre>}
          <PrimaryButton onClick={analyze} disabled={!selected} className="mt-4">分析失败原因</PrimaryButton>
          {analysis && <pre className="mt-4 max-h-[520px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{analysis}</pre>}
        </Card>
      </div>
    </>
  );
}
