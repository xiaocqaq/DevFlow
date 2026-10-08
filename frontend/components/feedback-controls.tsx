"use client";

import { Check, LoaderCircle, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { useEffect, useState } from "react";

import { apiPost } from "@/lib/api";

export type ChatFeedbackRecord = {
  id: string;
  repo_id: string;
  conversation_id: string;
  assistant_message_id: string;
  run_id?: string | null;
  rating: "helpful" | "unhelpful";
  reason?: FeedbackReason | null;
  comment?: string | null;
  review_status: "open" | "in_review" | "resolved" | "dismissed";
  notification_status: string;
  notification_error?: string | null;
};

type FeedbackReason =
  | "inaccurate"
  | "not_relevant"
  | "missing_context"
  | "unreliable_citation"
  | "tool_error"
  | "other";

type FeedbackControlsProps = {
  repoId: string;
  conversationId: string;
  messageId: string;
  initialFeedback?: ChatFeedbackRecord;
  onSaved: (feedback: ChatFeedbackRecord) => void;
};

const REASONS: Array<{ value: FeedbackReason; label: string }> = [
  { value: "inaccurate", label: "内容不准确" },
  { value: "not_relevant", label: "没有解决问题" },
  { value: "missing_context", label: "缺少关键上下文" },
  { value: "unreliable_citation", label: "引用不可靠" },
  { value: "tool_error", label: "工具执行异常" },
  { value: "other", label: "其他" }
];

export function FeedbackControls({
  repoId,
  conversationId,
  messageId,
  initialFeedback,
  onSaved
}: FeedbackControlsProps) {
  const [feedback, setFeedback] = useState(initialFeedback);
  const [panelOpen, setPanelOpen] = useState(false);
  const [reason, setReason] = useState<FeedbackReason | undefined>(initialFeedback?.reason ?? undefined);
  const [comment, setComment] = useState(initialFeedback?.comment ?? "");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setFeedback(initialFeedback);
    setReason(initialFeedback?.reason ?? undefined);
    setComment(initialFeedback?.comment ?? "");
  }, [initialFeedback]);

  async function submit(rating: "helpful" | "unhelpful") {
    setPending(true);
    setError("");
    try {
      const result = await apiPost<ChatFeedbackRecord>("/api/chat/feedback", {
        repo_id: repoId,
        conversation_id: conversationId,
        assistant_message_id: messageId,
        rating,
        reason: rating === "unhelpful" ? reason : null,
        comment: rating === "unhelpful" ? comment.trim() || null : null
      });
      setFeedback(result);
      setPanelOpen(false);
      onSaved(result);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "反馈提交失败");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="mt-3 border-t border-line pt-2">
      <div className="flex min-h-8 items-center gap-1">
        <button
          type="button"
          title="回答有帮助"
          aria-label="回答有帮助"
          aria-pressed={feedback?.rating === "helpful"}
          disabled={pending}
          onClick={() => void submit("helpful")}
          className={`flex h-8 w-8 items-center justify-center rounded-md transition disabled:cursor-not-allowed disabled:opacity-50 ${
            feedback?.rating === "helpful" ? "bg-emerald-50 text-emerald-700" : "text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          }`}
        >
          {pending && !panelOpen ? <LoaderCircle className="h-4 w-4 animate-spin" /> : <ThumbsUp className="h-4 w-4" />}
        </button>
        <button
          type="button"
          title="回答没帮助"
          aria-label="回答没帮助"
          aria-pressed={feedback?.rating === "unhelpful"}
          disabled={pending}
          onClick={() => {
            setPanelOpen(true);
            setError("");
          }}
          className={`flex h-8 w-8 items-center justify-center rounded-md transition disabled:cursor-not-allowed disabled:opacity-50 ${
            feedback?.rating === "unhelpful" ? "bg-rose-50 text-rose-700" : "text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          }`}
        >
          <ThumbsDown className="h-4 w-4" />
        </button>
        {feedback && !panelOpen && (
          <span className="ml-1 flex min-w-0 items-center gap-1 text-xs text-slate-500">
            <Check className="h-3.5 w-3.5 shrink-0 text-emerald-600" />
            已记录
            {feedback.run_id && <span className="truncate font-mono text-[11px]">trace {feedback.run_id.slice(0, 8)}</span>}
          </span>
        )}
      </div>

      {panelOpen && (
        <div className="mt-2 max-w-xl rounded-md border border-rose-100 bg-rose-50/60 p-3">
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs font-medium text-slate-700">这条回答哪里需要改进？</span>
            <button
              type="button"
              title="关闭"
              aria-label="关闭反馈"
              onClick={() => setPanelOpen(false)}
              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-slate-400 hover:bg-white hover:text-slate-700"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="mt-2 flex flex-wrap gap-2">
            {REASONS.map((item) => (
              <button
                key={item.value}
                type="button"
                aria-pressed={reason === item.value}
                onClick={() => setReason(item.value)}
                className={`rounded-md border px-2.5 py-1 text-xs transition ${
                  reason === item.value
                    ? "border-rose-300 bg-white text-rose-700"
                    : "border-slate-200 bg-white text-slate-600 hover:border-slate-300"
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
          <textarea
            value={comment}
            maxLength={1000}
            rows={2}
            onChange={(event) => setComment(event.target.value)}
            placeholder="补充说明（可选）"
            className="mt-2 w-full resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-700 outline-none focus:border-rose-300 focus:ring-2 focus:ring-rose-100"
          />
          <div className="mt-2 flex items-center justify-between gap-3">
            <span className="text-xs text-rose-600">{error}</span>
            <button
              type="button"
              disabled={pending}
              onClick={() => void submit("unhelpful")}
              className="flex h-8 shrink-0 items-center gap-1.5 rounded-md bg-slate-800 px-3 text-xs font-medium text-white hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {pending && <LoaderCircle className="h-3.5 w-3.5 animate-spin" />}
              提交反馈
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
