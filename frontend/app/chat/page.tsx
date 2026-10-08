"use client";

import { useEffect, useState } from "react";

import { Card, PageTitle, PrimaryButton } from "@/components/ui/card";
import { apiGet, apiPostStream } from "@/lib/api";
import type { Repository } from "@/lib/types";

const samples = [
  "Why did the latest CI fail?",
  "Summarize the risk in the latest PR",
  "Who should own the latest Issue?",
  "Generate this week's engineering report",
  "Read backend/app/services/agents/chat_agent.py and explain tool calling",
];

export default function ChatPage() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [repoId, setRepoId] = useState("");
  const [message, setMessage] = useState("Why did the latest CI fail?");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    apiGet<Repository[]>("/api/repos").then((items) => {
      setRepos(items);
      setRepoId((current) => current || items[0]?.id || "");
    });
  }, []);

  async function send() {
    setLoading(true);
    setResult("");
    try {
      await apiPostStream("/api/chat/stream", { repo_id: repoId, message, context: { time_range: "7d", messages: [] } }, (event, rawData) => {
        if (event === "delta") {
          const data = rawData as { content?: string };
          if (data.content) setResult((current) => current + data.content);
        }
        if (event === "final") {
          setResult(JSON.stringify(rawData, null, 2));
        }
      });
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <PageTitle
        title="Agent Chat"
        description="Use native tool calling to inspect code, search memory, run specialized analysis, and draft safe actions."
      />
      <Card>
        <select
          className="mb-3 w-full rounded-md border border-line bg-white px-3 py-2 text-sm"
          value={repoId}
          onChange={(event) => setRepoId(event.target.value)}
        >
          {repos.map((repo) => (
            <option key={repo.id} value={repo.id}>{repo.full_name}</option>
          ))}
        </select>
        <textarea
          className="h-32 w-full rounded-md border border-line bg-white px-3 py-3 text-sm leading-6 outline-none transition focus:border-teal-500 focus:ring-2 focus:ring-teal-100"
          value={message}
          onChange={(event) => setMessage(event.target.value)}
        />

        <div className="mt-3 flex flex-wrap gap-2 text-xs text-slate-500">
          {samples.map((sample) => (
            <button
              key={sample}
              onClick={() => setMessage(sample)}
              className="rounded-full border border-line bg-panel px-3 py-1.5 transition hover:border-teal-200 hover:bg-teal-50"
            >
              {sample}
            </button>
          ))}
        </div>

        <PrimaryButton onClick={send} disabled={loading || !repoId || !message.trim()} className="mt-4">
          {loading ? "Analyzing..." : "Ask Agent"}
        </PrimaryButton>

        {result && <pre className="mt-4 max-h-[560px] overflow-auto rounded-md border border-slate-800 bg-slate-950 p-4 text-xs text-slate-100">{result}</pre>}
      </Card>
    </>
  );
}
