"""全局配置管理模块.

本模块基于 Pydantic Settings 构建，负责统一管理系统运行所需的所有环境变量、
服务配置、模型参数及各三方集成的密钥与端点。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """系统全局配置类.

    【Python / Pydantic 语法要点】:
    1. 继承自 `BaseSettings` (来自 `pydantic_settings` 库):
       - Pydantic v2 将环境变量管理拆分至单独的 `pydantic-settings` 包。
       - 实例化时会自动从操作系统环境变量（Environment Variables）或指定的 `.env` 文件中读取同名字段。
       - 匹配规则默认大小写不敏感（如 `APP_ENV` 能自动映射到类属性 `app_env`）。
       - 具有强类型校验与自动类型转换能力（如字符串 "8000" 会自动转换为 int 类型 8000）。
    2. 类型注解规范 (Type Hinting):
       - 使用 Python 3.10+ 原生联合类型语法 `str | None`，等价于旧版 `typing.Optional[str]`。
       - 类属性定义的同时赋予默认值，若环境变量未提供该配置，则优雅降级使用默认值。
    3. `model_config = SettingsConfigDict(...)`:
       - Pydantic v2 取代旧版内部类 `class Config:` 的全新配置方式。
       - `env_file`: 接收元组指定环境变量文件路径，优先级从左到右加载。
       - `extra="ignore"`: 遇到环境变量中有额外未在当前类声明的键时，直接忽略而不抛出 ValidationError。
    """

    # --- 基础运行环境与 Web 服务配置 ---
    app_env: str = "development"
    app_name: str = "DevFlow AI"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

    # --- 数据库连接配置 (PostgreSQL + psycopg v3) ---
    database_url: str = "postgresql+psycopg://devflow:devflow@localhost:5432/devflow"

    # --- GitHub API 客户端配置 ---
    github_token: str | None = None
    github_api_base_url: str = "https://api.github.com"

    # --- LLM 大模型推理配置 (兼容 OpenAI 规范的 API) ---
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-v4-pro"
    rag_llm_enabled: bool = True

    # --- 文本向量嵌入 (Embedding) 配置 ---
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

    # --- 语义重排 (Rerank) 配置 ---
    rerank_api_key: str | None = None
    rerank_base_url: str | None = None
    qwen_rerank_model: str = "qwen3-rerank"
    qwen_retrieval_instruction: str = "Given a user question, retrieve passages that contain the facts needed to answer it."

    # --- 向量数据库 (Milvus) 配置 ---
    milvus_enabled: bool = True
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str | None = None
    milvus_collection: str = "devflow_rag_qwen1024"
    rag_rerank_top_n: int = 24

    # --- Ragas 评测系统配置 ---
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

    # --- 上下文窗口管理与动态压缩算法参数 ---
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

    # --- Agent 与工作流任务调度配置 ---
    workflow_task_timeout_seconds: float = 90.0
    prompt_cache_enabled: bool = True
    prompt_cache_key: str | None = "devflow-agent-v1"
    prompt_cache_retention: str | None = None
    repo_checkout_dir: str = ".data/repos"
    upload_dir: str = ".data/uploads"

    # --- 安全加密与 Webhook 回调配置 ---
    token_encryption_key: str = "change-me-in-local-dev"
    demo_mode: bool = True
    github_webhook_secret: str | None = None
    feishu_feedback_webhook_url: str | None = None
    feedback_trace_base_url: str | None = None

    # --- 代码库定时同步后台任务配置 ---
    auto_sync_enabled: bool = True
    auto_sync_interval_seconds: int = 300
    auto_sync_limit: int = 50

    model_config = SettingsConfigDict(env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore")


# 全局单例配置对象
# 【Python 设计模式要点】:
# 在模块级别直接实例化 `settings = Settings()`。
# 利用 Python 模块导入缓存机制（sys.modules），所有 `from app.core.config import settings`
# 的导入都会共享这同一个实例，实现轻量级、开箱即用的单例模式（Singleton Pattern）。
settings = Settings()
