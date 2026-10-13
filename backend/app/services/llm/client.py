"""LLM 大语言模型客户端适配模块.

本模块基于 httpx 异步网络库封装对兼容 OpenAI 接口规范的大模型
服务（如 DeepSeek、OpenAI、Qwen 等）的通信调用，支持 JSON 结构化输出、
同步文本生成及 Server-Sent Events (SSE) 流式传输。
"""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from app.core.config import settings
from app.services.context_compression import prompt_cache_model_kwargs


class LLMClient:
    """通用 LLM HTTP 异步客户端.

    【Python 语法要点 - 异步上下文与流式生成器】:
    1. `async with httpx.AsyncClient(...) as client`:
       异步网络连接池上下文管理器，请求完成后自动释放底层的 TCP 连接和套接字。
    2. `AsyncIterator[str]` 与 `async for ... in response.aiter_lines()`:
       异步生成器协议（PEP 525）。在网络数据块到达时通过 `yield` 逐块推送字符，
       客户端无需等待完整回复生成即可开始打字机渲染。
    3. `str.removeprefix("data: ")`:
       Python 3.9+ 原生字符串方法，安全剥离 SSE 协议头，较切片操作更具语义性。
    """

    def __init__(self) -> None:
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model

    async def chat_json(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        """请求大模型返回经过 JSON 解析的结构化字典.

        Args:
            system_prompt: 系统提示词，用于设定角色与输出格式约束。
            user_payload: 用户输入负载，会被自动序列化为 JSON 字符串。

        Returns:
            dict[str, Any]: 大模型输出的解析后字典；调用失败或解析错误时降级返回空字典 `{}`。
        """
        if not self.api_key:
            return {}

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
        }
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
                return {}

    async def chat_text(self, system_prompt: str, user_message: str) -> str:
        """非流式请求大模型获取纯文本回复.

        Args:
            system_prompt: 系统提示词。
            user_message: 用户原始文本消息。

        Returns:
            str: 模型生成的回答文本；失败时返回空字符串。
        """
        if not self.api_key:
            return ""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        }
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError:
                return ""
            return str(response.json()["choices"][0]["message"].get("content") or "")

    async def chat_text_stream(self, system_prompt: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """以 SSE 协议流式请求大模型输出，逐字块产生增量文本.

        【Python 语法要点 - 异步生成器 (Async Generator)】:
        - 函数签名返回 `AsyncIterator[str]`，内部使用 `yield` 产生数据；
        - 调用方需通过 `async for chunk in client.chat_text_stream(...)` 进行消费。

        Args:
            system_prompt: 系统提示词。
            messages: 对话历史消息列表。

        Yields:
            str: 模型的增量输出文本 Token。
        """
        if not self.api_key:
            return

        # 解包操作符 `*messages` 将历史消息拼接到列表
        payload_messages = [{"role": "system", "content": system_prompt}, *messages]
        payload: dict[str, Any] = {"model": self.model, "messages": payload_messages, "stream": True}
        payload.update(prompt_cache_model_kwargs())
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError:
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line.removeprefix("data: ").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield str(content)
