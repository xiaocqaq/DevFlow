import uuid
from types import SimpleNamespace

from app.services.chat_memory import ProgressiveContextManager
from app.services.context_compression import (
    MICROCOMPACT_MARKER,
    adjust_keep_start_to_preserve_api_invariants,
    calculate_context_pressure,
    compress_messages,
    compress_retrieval_results,
    ensure_compaction_summary_preserves_key_facts,
    estimate_tokens,
    fallback_memory_update,
    messages_after_compact_boundary,
    merge_memory,
    microcompact_messages,
)


def test_compress_messages_keeps_recent_turns_under_budget() -> None:
    messages = [
        {"role": "user", "content": "old setup note " * 200},
        {"role": "assistant", "content": "old answer " * 200},
        {"role": "user", "content": "What did we decide about auth middleware?"},
    ]

    compressed, stats = compress_messages(messages, max_tokens=120)

    assert compressed[-1]["content"] == "What did we decide about auth middleware?"
    assert stats["kept_messages"] < stats["input_messages"]
    assert stats["omitted_messages"] > 0
    assert stats["exact_suffix_messages"] >= 1
    assert estimate_tokens("\n".join(item["content"] for item in compressed)) <= 140


def test_retrieval_compression_prefers_query_relevant_sentence() -> None:
    results = compress_retrieval_results(
        "auth token refresh middleware",
        [
            {
                "id": "auth",
                "title": "auth docs",
                "source_type": "workspace_file",
                "source_id": "auth",
                "score": 1.0,
                "_raw_content": "billing copy. " * 100 + "The auth middleware refreshes expired tokens before validating the session.",
            }
        ],
        max_tokens_per_item=40,
    )

    assert "auth middleware refreshes expired tokens" in results[0]["snippet"]
    assert results[0]["compression"]["compressed_estimated_tokens"] < results[0]["compression"]["original_estimated_tokens"]
    assert "_raw_content" not in results[0]


def test_retrieval_compression_respects_global_token_budget() -> None:
    results = compress_retrieval_results(
        "auth token",
        [
            {
                "id": f"doc-{index}",
                "score": 1.0 - index * 0.1,
                "_raw_content": ("auth token refresh evidence. " * 80),
            }
            for index in range(5)
        ],
        max_tokens_per_item=40,
        max_total_tokens=75,
    )

    assert sum(item["compression"]["compressed_estimated_tokens"] for item in results) <= 75
    assert all(item["retrieval"]["selected_for_context"] is True for item in results)
    assert len(results) < 5


def test_memory_update_merges_structured_sections() -> None:
    update = fallback_memory_update(
        "I prefer concise answers. Can we keep token refresh in middleware?",
        "Decision: keep token refresh in middleware. Next step: add tests.",
        "direct_answer",
        [],
    )
    merged = merge_memory(
        {"summary": "", "facts": [], "decisions": [], "open_questions": [], "tasks": [], "user_preferences": [], "repo_context": [], "citations": []},
        update,
        update["summary"],
        max_tokens=180,
    )

    assert merged["decisions"]
    assert merged["tasks"]
    assert merged["user_preferences"]


def test_pressure_uses_effective_window_stages() -> None:
    pressure = calculate_context_pressure(116_000, model="gpt-4o-mini")

    assert pressure.stage in {"auto_compact", "aggressive", "manual_path"}
    assert pressure.effective_window_tokens > 100_000
    assert pressure.remaining_tokens >= 0


def test_microcompact_clears_old_regenerable_tool_context() -> None:
    messages = [
        {
            "role": "tool_result",
            "content": "irrelevant setup. " * 300 + "The auth middleware refreshes expired tokens before validation.",
            "route": "workspace.read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "arguments": {}, "status": "success"}],
        },
        {"role": "user", "content": "What did we decide about auth middleware?"},
        {"role": "assistant", "content": "Keep token refresh in middleware."},
    ]

    compacted, stats = microcompact_messages(messages, "auth middleware", keep_recent=2, max_excerpt_tokens=50)

    assert MICROCOMPACT_MARKER in compacted[0]["content"]
    assert "auth middleware refreshes expired tokens" in compacted[0]["content"]
    assert compacted[-1]["content"] == "Keep token refresh in middleware."
    assert stats["compacted_messages"] == 1
    assert stats["estimated_tokens_saved"] > 0


def test_compaction_summary_preserves_paths_errors_and_todos() -> None:
    messages = [
        {
            "role": "user",
            "content": "Decision: keep token refresh in middleware.\nNext step: run pytest backend/app/tests/test_auth.py.",
        },
        {
            "role": "assistant",
            "content": "backend/app/auth.py raised TimeoutError: token refresh failed.",
        },
    ]

    summary, report = ensure_compaction_summary_preserves_key_facts(
        messages,
        "We discussed auth middleware.",
        max_tokens=180,
    )

    assert "backend/app/auth.py" in summary
    assert "pytest backend/app/tests/test_auth.py" in summary
    assert "TimeoutError" in summary
    assert report["appended_missing_count"] >= 3
    assert report["missing_count"] == 0


def test_messages_after_compact_boundary_uses_latest_boundary() -> None:
    messages = [
        {"role": "user", "content": "old"},
        {"role": "system", "content": "first", "meta": {"subtype": "compact_boundary"}},
        {"role": "assistant", "content": "middle"},
        {"role": "system", "content": "latest", "meta": {"subtype": "compact_boundary"}},
        {"role": "user", "content": "current"},
    ]

    sliced = messages_after_compact_boundary(messages)

    assert [item["content"] for item in sliced] == ["latest", "current"]


def test_keep_start_moves_back_to_preserve_tool_use_result_pair() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "calling read",
            "tool_calls": [{"id": "call_read", "tool_name": "workspace.read_file"}],
        },
        {"role": "tool_result", "content": "file contents", "tool_call_id": "call_read"},
        {"role": "user", "content": "continue"},
    ]

    assert adjust_keep_start_to_preserve_api_invariants(messages, 1) == 0


def test_formal_compaction_keep_rows_preserves_tool_pair() -> None:
    manager = ProgressiveContextManager(db=None)  # type: ignore[arg-type]
    rows = [
        SimpleNamespace(
            id=uuid.uuid4(),
            role="assistant",
            content="calling read",
            route=None,
            tool_calls=[{"id": "call_read"}],
            meta={},
        ),
        SimpleNamespace(
            id=uuid.uuid4(),
            role="tool_result",
            content="file contents",
            route=None,
            tool_calls=[],
            meta={"tool_call_id": "call_read"},
        ),
        SimpleNamespace(id=uuid.uuid4(), role="user", content="continue", route=None, tool_calls=[], meta={}),
    ]

    kept = manager._messages_to_keep(rows, keep_count=2)  # type: ignore[arg-type]

    assert kept == rows


def test_microcompact_never_rewrites_assistant_final_answer() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Final answer based on workspace.read_file. " * 300,
            "route": "workspace.read_file",
            "tool_calls": [{"tool_name": "workspace.read_file", "status": "success"}],
        },
        {"role": "user", "content": "Continue."},
    ]

    compacted, stats = microcompact_messages(messages, "continue", keep_recent=1)

    assert compacted[0]["content"] == messages[0]["content"]
    assert stats["compacted_messages"] == 0
