"""GitHub REST API 客户端模块.

本模块基于 httpx 异步库封装对 GitHub API 的网络请求交互，
处理认证凭证注入、重定向跟随（Redirect Following）以及常用 Issue、PR 与日志查询。
"""

from typing import Any

import httpx

from app.core.config import settings


class GitHubClient:
    """GitHub 官方 REST API 异步客户端封装.

    【Python 语法要点 - 异步 HTTP 请求与重定向处理】:
    - `follow_redirects=True`: 允许 HTTP 301/302 自动重定向追踪，这在仓库改名、转让所有权或下载大型归档包时必不可少；
    - `response.raise_for_status()`: 当 HTTP 状态码为 4xx（如 401 Unauthorized、404 Not Found）或 5xx 时自动引发 `HTTPStatusError`；
    - `response.content` 返回原始二进制字节串（bytes，如下载构建日志或工作流产物），`response.json()` 自动反序列化 JSON 响应。
    """

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self.token = token or settings.github_token
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        """构造 GitHub API 请求头（注入版本声明与 Bearer Token）."""
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """发送异步 GET 请求并解析 JSON 响应."""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.json()

    async def get_bytes(self, path: str, params: dict[str, Any] | None = None) -> bytes:
        """发送异步 GET 请求并返回原始二进制响应体（用于下载压缩包与日志）."""
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(f"{self.base_url}{path}", headers=self._headers(), params=params)
            response.raise_for_status()
            return response.content

    async def post(self, path: str, payload: dict[str, Any]) -> Any:
        """发送异步 POST 请求并解析 JSON 响应."""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.post(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def patch(self, path: str, payload: dict[str, Any]) -> Any:
        """发送异步 PATCH 部分更新请求并解析 JSON 响应."""
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.patch(f"{self.base_url}{path}", headers=self._headers(), json=payload)
            response.raise_for_status()
            return response.json()

    async def get_repo(self, owner: str, repo: str) -> dict[str, Any]:
        """获取指定仓库元数据信息."""
        return await self.get(f"/repos/{owner}/{repo}")

    async def create_issue_comment(self, owner: str, repo: str, issue_number: int, body: str) -> dict[str, Any]:
        """在指定 Issue 下发表评论."""
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/comments", {"body": body})

    async def create_issue(self, owner: str, repo: str, title: str, body: str) -> dict[str, Any]:
        """在仓库中新建 Issue."""
        return await self.post(f"/repos/{owner}/{repo}/issues", {"title": title, "body": body})

    async def close_issue(self, owner: str, repo: str, issue_number: int) -> dict[str, Any]:
        """关闭指定编号的 Issue."""
        return await self.patch(f"/repos/{owner}/{repo}/issues/{issue_number}", {"state": "closed"})

    async def add_issue_labels(self, owner: str, repo: str, issue_number: int, labels: list[str]) -> dict[str, Any]:
        """为指定 Issue 批量打上标签."""
        return await self.post(f"/repos/{owner}/{repo}/issues/{issue_number}/labels", {"labels": labels})
