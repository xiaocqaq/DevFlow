import re
from typing import Any

from app.schemas.analysis import IssueAnalysisOutput, IssueDrafts
from app.services.agents.base import BaseAgent
from app.services.llm.client import LLMClient
from app.services.llm.prompts import ISSUE_ANALYSIS_PROMPT


class IssueAgent(BaseAgent):
    name = "issue_analyst_agent"
    description = "分析 GitHub Issue 的分诊结论、优先级、复杂度、负责人、证据和下一步动作。"
    available_tools = ["get_issue_detail", "search_similar_issues", "search_related_code", "search_project_docs", "read_thread_context"]

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        fallback = self._fallback_analysis(input_data, context)
        llm_output = await self.llm.chat_json(ISSUE_ANALYSIS_PROMPT, {"issue": input_data, "context": context})
        if llm_output:
            try:
                normalized = self._merge_with_fallback(llm_output, fallback, input_data, context)
                return IssueAnalysisOutput.model_validate(normalized).model_dump()
            except Exception:
                pass
        return fallback

    def _fallback_analysis(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        text = f"{input_data.get('title', '')} {input_data.get('body', '')}".lower()
        body = str(input_data.get("body") or "")
        labels = [label.lower() for label in input_data.get("labels", [])]
        category = self._classify_category(text, labels)
        priority = self._classify_priority(text, labels)
        complexity = self._estimate_complexity(text, labels)
        owner, owner_reason = self._suggest_owner(input_data, text, context)
        duplicates = self._validated_duplicates(context.get("duplicate_candidates", []))
        conclusion, conclusion_reason = self._choose_conclusion(input_data, text, labels, category, complexity, duplicates)
        checklist = self._build_checklist(conclusion, category)
        drafts = self._build_drafts(input_data, conclusion, checklist)
        evidence = self._collect_evidence(input_data, context)
        confidence = self._estimate_confidence(body, evidence, duplicates, owner)
        return IssueAnalysisOutput(
            summary=self._build_summary(input_data, conclusion),
            conclusion=conclusion,
            conclusion_reason=conclusion_reason,
            category=category,
            priority=priority,
            complexity=complexity,
            suggested_owner=owner,
            owner_reason=owner_reason,
            duplicate_candidates=duplicates,
            evidence=evidence,
            checklist=checklist,
            action_items=checklist,
            drafts=drafts,
            confidence=confidence,
        ).model_dump()

    def _merge_with_fallback(self, output: dict[str, Any], fallback: dict[str, Any], input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        merged = {**fallback, **output}
        if not merged.get("checklist") and merged.get("action_items"):
            merged["checklist"] = merged["action_items"]
        if not merged.get("action_items") and merged.get("checklist"):
            merged["action_items"] = merged["checklist"]
        if not merged.get("evidence"):
            merged["evidence"] = fallback["evidence"]
        merged["evidence"] = self._dedupe_evidence(merged.get("evidence"))
        if not merged.get("duplicate_candidates"):
            merged["duplicate_candidates"] = fallback["duplicate_candidates"]
        if not isinstance(merged.get("drafts"), dict):
            merged["drafts"] = fallback["drafts"]
        if not self._is_allowed_owner(str(merged.get("suggested_owner") or ""), input_data, context):
            merged["suggested_owner"] = fallback["suggested_owner"]
            merged["owner_reason"] = fallback["owner_reason"]
        if self._should_prefer_fallback_owner(str(merged.get("suggested_owner") or ""), str(fallback.get("suggested_owner") or "")):
            merged["suggested_owner"] = fallback["suggested_owner"]
            merged["owner_reason"] = fallback["owner_reason"]
        if merged.get("category") == "Question" and fallback.get("category") in {"Bug", "Feature", "Documentation", "Ops", "Refactor", "Test"}:
            merged["category"] = fallback["category"]
        return merged

    def _should_prefer_fallback_owner(self, current_owner: str, fallback_owner: str) -> bool:
        generic_owners = {"", "待分配", "unassigned", "backend", "frontend", "ci", "docs"}
        return fallback_owner not in generic_owners and current_owner.lower() in generic_owners

    def _is_allowed_owner(self, owner: str, input_data: dict[str, Any], context: dict[str, Any]) -> bool:
        normalized_owner = owner.strip().lower()
        if normalized_owner in {"", "待分配", "unassigned"}:
            return True
        allowed = {str(item).strip().lower() for item in input_data.get("assignees") or [] if str(item).strip()}
        members = context.get("team_members") or []
        if isinstance(members, list):
            for member in members:
                if isinstance(member, dict) and str(member.get("name") or "").strip():
                    allowed.add(str(member.get("name")).strip().lower())
        return normalized_owner in allowed

    def _build_summary(self, input_data: dict[str, Any], conclusion: str) -> str:
        number = input_data.get("number")
        title = str(input_data.get("title") or "Untitled Issue").strip()
        prefix = f"Issue #{number}" if number else "Issue"
        return f"{prefix} {title} 的建议结论是：{conclusion}。"

    def _classify_category(self, text: str, labels: list[str]) -> str:
        joined = " ".join(labels) + " " + text
        rules = [
            ("Bug", ["bug", "error", "fail", "500", "crash", "exception", "broken", "regression"]),
            ("Documentation", ["doc", "documentation", "readme", "guide", "文档", "说明"]),
            ("Test", ["test", "pytest", "coverage", "单测", "测试"]),
            ("Ops", ["ops", "ci", "workflow", "deploy", "docker", "infra", "环境", "权限"]),
            ("Refactor", ["refactor", "cleanup", "重构", "technical debt"]),
            ("Feature", ["feature", "enhancement", "proposal", "需求", "新增"]),
            ("Question", ["question", "how to", "why does", "咨询", "疑问"]),
        ]
        for category, tokens in rules:
            if any(token in joined for token in tokens):
                return category
        return "Feature"

    def _classify_priority(self, text: str, labels: list[str]) -> str:
        joined = " ".join(labels) + " " + text
        if any(token in joined for token in ["p0", "sev0", "critical", "security", "data loss", "数据丢失", "全量不可用"]):
            return "P0"
        if any(token in joined for token in ["p1", "sev1", "crash", "cannot login", "500", "生产", "阻塞", "blocker"]):
            return "P1"
        if any(token in joined for token in ["p3", "minor", "polish", "typo", "文案", "低优"]):
            return "P3"
        return "P2"

    def _estimate_complexity(self, text: str, labels: list[str]) -> str:
        joined = " ".join(labels) + " " + text
        if any(token in joined for token in ["xl", "architecture", "migration", "rewrite", "跨模块", "架构"]):
            return "XL"
        if any(token in joined for token in ["large", "complex", "integration", "schema", "多模块"]):
            return "L"
        if len(text) < 280 and any(token in joined for token in ["typo", "copy", "文案", "doc"]):
            return "S"
        return "M" if len(text) < 1400 else "L"

    def _suggest_owner(self, input_data: dict[str, Any], issue_text: str, context: dict[str, Any]) -> tuple[str, str]:
        assignees = input_data.get("assignees") or []
        if assignees:
            owner = str(assignees[0])
            return owner, f"Issue 已分配给 {owner}，优先沿用现有负责人。"

        signal_text = self._owner_signal_text(issue_text, context)
        team_owner, team_reason = self._suggest_owner_from_team(issue_text, signal_text, context)
        if team_owner:
            return team_owner, team_reason or "团队画像与 Issue 关键词匹配。"

        area = self._infer_area(signal_text)
        members = context.get("team_members") or []
        if not isinstance(members, list) or not members:
            if area != "待分配":
                return "待分配", f"当前项目还没有添加团队成员；这个 Issue 更像 {area} 方向，请先在团队面板添加对应开发成员后再分配。"
            return "待分配", "当前项目还没有添加团队成员，请先在团队面板添加开发成员后再分配。"
        if area != "待分配":
            return "待分配", f"未找到与 {area} 方向匹配的团队成员，请先补充团队成员画像后再分配。"
        return "待分配", "当前 Issue 与团队画像、代码/文档证据的匹配度不足，需要人工确认负责人。"

    def _suggest_owner_from_team(self, issue_text: str, signal_text: str, context: dict[str, Any]) -> tuple[str | None, str | None]:
        members = context.get("team_members") or []
        if not isinstance(members, list):
            return None, None

        best_name = None
        best_score = 0
        best_reasons: list[str] = []
        for member in members:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            profile = " ".join(
                str(member.get(field) or "").lower()
                for field in ["role", "strengths", "techStack", "tech_stack"]
            )
            score = 0
            reasons: list[str] = []
            for term in self._profile_terms(profile):
                if len(term) >= 2 and term in signal_text:
                    score += 1
                    reasons.append(term)
            for domain, issue_terms, profile_terms in self._owner_domain_rules():
                if any(term in signal_text for term in issue_terms) and any(term in profile for term in profile_terms):
                    score += 8
                    reasons.append(domain)
            if name and score > best_score:
                best_name = name
                best_score = score
                best_reasons = reasons
        if best_name and best_score > 0:
            reason_text = "、".join(dict.fromkeys(best_reasons[:4])) or "团队画像"
            return best_name, f"团队成员画像与 Issue/代码/文档信号匹配：{reason_text}，匹配分 {best_score}。"
        return None, None

    def _infer_area(self, issue_text: str) -> str:
        area_keywords = {
            "backend": ["api", "database", "token", "server", "fastapi", "postgres", "auth", "jwt"],
            "frontend": ["ui", "page", "button", "react", "next", "css", "layout", "browser", "页面", "按钮", "electron", "desktop", "桌面化", "桌面端"],
            "ci": ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy"],
            "docs": ["readme", "doc", "文档", "说明"],
        }
        for area, keywords in area_keywords.items():
            if any(keyword in issue_text for keyword in keywords):
                return area
        return "待分配"

    def _owner_signal_text(self, issue_text: str, context: dict[str, Any]) -> str:
        parts = [issue_text]
        for key in ["code_references", "project_documents", "conversation_evidence"]:
            items = context.get(key)
            if not isinstance(items, list):
                continue
            for item in items[:6]:
                if isinstance(item, dict):
                    parts.extend([str(item.get("title") or ""), str(item.get("snippet") or ""), str(item.get("content") or "")])
        text = " ".join(parts).lower()
        if any(term in text for term in ["桌面化", "桌面端", "桌面应用", "electron"]):
            text += " electron desktop windows macos javascript node frontend"
        return text

    def _profile_terms(self, profile: str) -> list[str]:
        normalized = profile.replace("擅长", " ").replace("熟悉", " ").replace("技术栈", " ")
        return re.findall(r"[a-zA-Z0-9_+#.\u4e00-\u9fff]+", normalized.lower())

    def _owner_domain_rules(self) -> list[tuple[str, list[str], list[str]]]:
        return [
            (
                "桌面化/electron",
                ["桌面化", "桌面端", "桌面应用", "electron", "desktop", "windows", "macos", "native app"],
                ["electron", "desktop", "桌面", "windows", "macos", "tauri", "javascript", "js", "node", "frontend", "前端"],
            ),
            (
                "后端/API",
                ["api", "database", "token", "server", "fastapi", "postgres", "auth", "jwt"],
                ["api", "database", "server", "fastapi", "postgres", "auth", "jwt", "后端"],
            ),
            (
                "前端/UI",
                ["ui", "page", "button", "react", "next", "css", "layout", "browser", "页面", "按钮"],
                ["ui", "react", "next", "css", "browser", "frontend", "前端"],
            ),
            (
                "CI/工程化",
                ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy"],
                ["ci", "workflow", "actions", "pytest", "lint", "build", "deploy", "工程化"],
            ),
        ]

    def _validated_duplicates(self, candidates: Any) -> list[dict[str, Any]]:
        if not isinstance(candidates, list):
            return []
        rows = []
        for item in candidates[:5]:
            if not isinstance(item, dict):
                continue
            issue_id = str(item.get("issue_id") or item.get("source_id") or "").strip()
            title = str(item.get("title") or "").strip()
            try:
                score = max(0.0, min(1.0, float(item.get("score") or 0)))
            except (TypeError, ValueError):
                score = 0.0
            if issue_id and title:
                rows.append({"issue_id": issue_id, "title": title, "score": score})
        return rows

    def _choose_conclusion(
        self,
        input_data: dict[str, Any],
        text: str,
        labels: list[str],
        category: str,
        complexity: str,
        duplicates: list[dict[str, Any]],
    ) -> tuple[str, str]:
        state = str(input_data.get("state") or "").lower()
        joined = " ".join(labels) + " " + text
        if state == "closed" or any(token in joined for token in ["invalid", "wontfix", "won't fix", "not planned", "无效", "不处理"]):
            return "关闭", "Issue 已关闭或带有无效/不处理信号，建议归档并保留关闭说明。"
        if any(float(item.get("score") or 0) >= 0.86 for item in duplicates):
            return "合并到已有 Issue", "相似历史 Issue 的匹配度较高，建议合并上下文，避免重复推进。"
        if complexity in {"XL"} or self._looks_like_multi_scope(text):
            return "拆分", "Issue 范围偏大或包含多个独立目标，拆分后更容易排期和验收。"
        if self._needs_clarification(text, category):
            return "等待澄清", "当前描述缺少影响范围、复现/验收条件或期望结果，直接进入开发风险较高。"
        return "进入开发", "Issue 信息基本可执行，优先确认验收口径后进入开发。"

    def _looks_like_multi_scope(self, text: str) -> bool:
        separators = text.count("\n-") + text.count("\n*") + text.count("、") + text.count(";")
        broad_terms = ["同时", "以及", "并且", "多个", "all of", "migration", "architecture", "跨模块"]
        return separators >= 5 or any(term in text for term in broad_terms)

    def _needs_clarification(self, text: str, category: str) -> bool:
        if len(text.strip()) < 80:
            return True
        if category == "Bug":
            has_repro = any(token in text for token in ["reproduce", "steps", "复现", "步骤", "expected", "actual", "期望", "实际"])
            has_impact = any(token in text for token in ["impact", "影响", "scope", "范围", "blocking", "阻塞"])
            return not (has_repro or has_impact)
        if category == "Feature":
            has_acceptance = any(token in text for token in ["acceptance", "验收", "criteria", "why", "value", "价值", "场景"])
            return not has_acceptance and len(text.strip()) < 220
        return False

    def _collect_evidence(self, input_data: dict[str, Any], context: dict[str, Any]) -> list[dict[str, Any]]:
        evidence = [
            {
                "source_type": "issue",
                "title": f"Issue #{input_data.get('number')}: {input_data.get('title')}",
                "snippet": self._clip("\n".join(filter(None, [str(input_data.get("title") or ""), str(input_data.get("body") or "")])), 320),
                "reference": str(input_data.get("id") or input_data.get("number") or ""),
                "confidence": 1.0,
            }
        ]
        for duplicate in self._validated_duplicates(context.get("duplicate_candidates", []))[:3]:
            evidence.append(
                {
                    "source_type": "similar_issue",
                    "title": duplicate["title"],
                    "snippet": f"相似度 {duplicate['score']:.2f}，可作为重复/历史处理经验参考。",
                    "reference": duplicate["issue_id"],
                    "confidence": duplicate["score"],
                }
            )
        for key, source_type in [
            ("code_references", "code"),
            ("project_documents", "document"),
            ("conversation_evidence", "conversation"),
        ]:
            evidence.extend(self._evidence_from_context(context.get(key), source_type))
        evidence.extend(self._team_evidence(context.get("team_members")))
        return self._dedupe_evidence(evidence)[:10]

    def _dedupe_evidence(self, items: Any) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str, str]] = set()
        for item in items:
            if not isinstance(item, dict) or not item.get("snippet"):
                continue
            key = (
                str(item.get("source_type") or ""),
                str(item.get("reference") or ""),
                str(item.get("title") or ""),
                str(item.get("snippet") or "")[:180],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
        return rows

    def _evidence_from_context(self, items: Any, default_source_type: str) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            return []
        rows = []
        for item in items[:4]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("path") or item.get("source_id") or default_source_type).strip()
            snippet = self._clip(str(item.get("snippet") or item.get("content") or ""), 320)
            if not snippet:
                continue
            score = item.get("score")
            try:
                confidence = max(0.0, min(1.0, float(score))) if score is not None else None
            except (TypeError, ValueError):
                confidence = None
            rows.append(
                {
                    "source_type": str(item.get("source_type") or default_source_type),
                    "title": title,
                    "snippet": snippet,
                    "reference": str(item.get("reference") or item.get("source_id") or item.get("id") or ""),
                    "confidence": confidence,
                }
            )
        return rows

    def _team_evidence(self, members: Any) -> list[dict[str, Any]]:
        if not isinstance(members, list) or not members:
            return []
        rows = []
        for member in members[:3]:
            if not isinstance(member, dict):
                continue
            name = str(member.get("name") or "").strip()
            if not name:
                continue
            profile = "；".join(
                str(member.get(field) or "").strip()
                for field in ["role", "strengths", "techStack", "tech_stack"]
                if str(member.get(field) or "").strip()
            )
            rows.append(
                {
                    "source_type": "team_member",
                    "title": f"团队成员：{name}",
                    "snippet": profile or "团队成员资料可作为负责人分配依据。",
                    "reference": name,
                    "confidence": None,
                }
            )
        return rows

    def _build_checklist(self, conclusion: str, category: str) -> list[str]:
        if conclusion == "等待澄清":
            return [
                "请补充问题背景、影响范围和用户场景。",
                "Bug 类问题补充复现步骤、期望结果、实际结果和日志/截图。",
                "Feature 类需求补充验收标准和非目标范围。",
                "澄清后重新运行 Issue Agent 分析。",
            ]
        if conclusion == "合并到已有 Issue":
            return [
                "确认相似 Issue 是否覆盖当前诉求。",
                "把当前 Issue 的新增信息整理为评论追加到已有 Issue。",
                "为当前 Issue 标记 duplicate/关联关系。",
                "经人工确认后关闭当前重复 Issue。",
            ]
        if conclusion == "拆分":
            return [
                "按用户价值、模块边界或交付顺序拆成 2-5 个子任务。",
                "为每个子任务补充验收标准、依赖关系和建议负责人。",
                "确认最小可交付范围，优先排第一个可独立验证的子任务。",
                "把拆分说明作为评论回填到原 Issue。",
            ]
        if conclusion == "关闭":
            return [
                "确认关闭原因是无效、重复、已完成还是不计划处理。",
                "补充一条简短关闭评论，说明依据和替代路径。",
                "经人工确认后更新标签或关闭 Issue。",
            ]
        baseline = [
            "确认 Issue 验收标准和不做范围。",
            "定位相关代码/文档位置并补充设计备注。",
            "创建开发分支并实现最小修复/功能增量。",
            "补充自动化测试或手工验证步骤。",
            "提交 PR 并在描述中关联该 Issue。",
        ]
        if category == "Bug":
            baseline.insert(2, "先写一个能复现问题的失败测试或最小复现脚本。")
        return baseline

    def _build_drafts(self, input_data: dict[str, Any], conclusion: str, checklist: list[str]) -> IssueDrafts:
        number = input_data.get("number")
        title = str(input_data.get("title") or "该 Issue")
        clarification = None
        task_breakdown = None
        if conclusion == "等待澄清":
            clarification = (
                f"针对 Issue #{number} {title}，目前信息还不足以直接进入开发。"
                "请补充：影响范围、复现步骤/使用场景、期望结果、实际结果，以及是否有日志、截图或相关版本信息。"
            )
        if conclusion in {"拆分", "进入开发"}:
            task_breakdown = "建议下一步拆解：\n" + "\n".join(f"- {item}" for item in checklist[:5])
        if conclusion == "合并到已有 Issue":
            task_breakdown = (
                f"建议将 Issue #{number} 的新增信息合并到相似历史 Issue，并在当前 Issue 留下 duplicate 说明后再关闭。"
            )
        return IssueDrafts(clarification_comment=clarification, task_breakdown=task_breakdown)

    def _estimate_confidence(self, body: str, evidence: list[dict[str, Any]], duplicates: list[dict[str, Any]], owner: str) -> float:
        score = 0.45
        if len(body.strip()) >= 120:
            score += 0.12
        if len(evidence) >= 3:
            score += 0.14
        if duplicates:
            score += 0.08
        if owner != "待分配":
            score += 0.08
        return round(max(0.35, min(0.86, score)), 2)

    def _clip(self, text: str, limit: int) -> str:
        text = (text or "").strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
