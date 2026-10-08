from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    app_name: str = "DevFlow AI"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

    database_url: str = "postgresql+psycopg://devflow:devflow@localhost:5432/devflow"

    github_token: str | None = None
    github_api_base_url: str = "https://api.github.com"

    llm_api_key: str | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-pro"
    rag_llm_enabled: bool = True
    embedding_provider: str = "qwen_local"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dimensions: int = 1024
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    local_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    local_embedding_device: str | None = None
    qwen_embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    dashscope_api_key: str | None = None
    dashscope_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    rerank_api_key: str | None = None
    rerank_base_url: str | None = None
    qwen_rerank_model: str = "qwen3-rerank"
    qwen_retrieval_instruction: str = "Given a user question, retrieve passages that contain the facts needed to answer it."

    milvus_enabled: bool = True
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_collection: str = "devflow_rag_qwen1024"
    rag_rerank_top_n: int = 24
    ragas_enabled: bool = True
    ragas_judge_api_key: str | None = None
    ragas_judge_base_url: str | None = None
    ragas_judge_model: str | None = None
    ragas_timeout_seconds: float = 120.0
    ragas_judge_max_tokens: int = 8192
    ragas_answer_relevancy_strictness: int = 1
    ragas_max_cases: int = 5
    ragas_context_precision_min: float = 0.75
    ragas_context_recall_min: float = 0.80
    ragas_faithfulness_min: float = 0.90
    ragas_answer_relevancy_min: float = 0.80
    ragas_agent_goal_accuracy_min: float = 0.80
    ragas_tool_call_accuracy_min: float = 0.90
    ragas_tool_call_f1_min: float = 0.90
    context_max_input_tokens: int = 24000
    context_reserved_response_tokens: int = 3000
    context_dynamic_model_window_enabled: bool = True
    context_model_window_tokens: int | None = None
    context_compact_reserved_output_tokens: int = 12000
    context_auto_compact_enabled: bool = True
    context_warning_buffer_tokens: int = 20000
    context_auto_compact_buffer_tokens: int = 13000
    context_aggressive_buffer_tokens: int = 8000
    context_manual_buffer_tokens: int = 3000
    context_max_compaction_failures: int = 3
    context_microcompact_keep_recent_messages: int = 8
    context_compact_keep_recent_messages: int = 8
    context_compact_summary_tokens: int = 1800
    context_system_ratio: float = 0.12
    context_memory_ratio: float = 0.22
    context_evidence_ratio: float = 0.34
    context_recent_ratio: float = 0.22
    context_tool_ratio: float = 0.10
    workflow_task_timeout_seconds: float = 90.0
    prompt_cache_enabled: bool = True
    prompt_cache_key: str | None = "devflow-agent-v1"
    prompt_cache_retention: str | None = None
    repo_checkout_dir: str = ".data/repos"
    upload_dir: str = ".data/uploads"

    token_encryption_key: str = "change-me-in-local-dev"
    demo_mode: bool = True
    github_webhook_secret: str | None = None
    feishu_feedback_webhook_url: str | None = None
    feedback_trace_base_url: str | None = None
    auto_sync_enabled: bool = True
    auto_sync_interval_seconds: int = 300
    auto_sync_limit: int = 50

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")


settings = Settings()
