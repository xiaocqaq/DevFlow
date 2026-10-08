from typing import Any

from app.services.agents.base import BaseAgent


class ReportAgent(BaseAgent):
    name = "report_agent"
    description = "根据仓库活动生成工程周报。"
    available_tools = ["list_issues_by_time_range", "list_prs_by_time_range", "list_failed_workflow_runs"]

    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        issues = input_data.get("issues", [])
        prs = input_data.get("pull_requests", [])
        ci_runs = input_data.get("workflow_runs", [])
        metrics = input_data.get("metrics", {})
        failed_ci = [run for run in ci_runs if run.get("conclusion") == "failure"]
        merged_prs = [pr for pr in prs if pr.get("merged_at")]
        open_prs = [pr for pr in prs if pr.get("state") == "open"]
        closed_issues = [issue for issue in issues if issue.get("state") == "closed"]
        repo_name = context.get("repo_name") or f"Repo {context.get('repo_id', '')}".strip()
        date_range = f"{context.get('start_date', '')} 至 {context.get('end_date', '')}".strip()
        merged_pr_numbers = ", ".join(f"#{pr.get('number')}" for pr in merged_prs[:8]) or "暂无"
        closed_issue_numbers = ", ".join(f"#{issue.get('number')}" for issue in closed_issues[:8]) or "暂无"
        markdown = (
            f"# {repo_name} 研发周报\n\n"
            f"> 时间范围：{date_range}\n\n"
            "## 数据概览\n"
            f"- Issues：{metrics.get('issues', len(issues))} 个\n"
            f"- PRs：{metrics.get('pull_requests', len(prs))} 个，其中 Merged PR {metrics.get('merged_prs', len(merged_prs))} 个，Open PR {metrics.get('open_prs', len(open_prs))} 个\n"
            f"- 失败 CI：{metrics.get('failed_ci', len(failed_ci))} 次\n\n"
            "## 本周完成\n"
            f"- 合并 PR {len(merged_prs)} 个：{merged_pr_numbers}。\n"
            f"- 关闭 Issue {len(closed_issues)} 个：{closed_issue_numbers}。\n\n"
            "## 进行中\n"
            f"- 当前仍有 {len(open_prs)} 个 Open PR 需要跟进。\n"
            + "\n".join(f"- #{pr.get('number')} {pr.get('title')}" for pr in open_prs[:5])
            + ("\n" if open_prs else "- 暂无 Open PR。\n")
            + "\n"
            "## 风险与阻塞\n"
            f"- 失败 CI {len(failed_ci)} 次，建议优先排查重复失败的 workflow。\n"
            + ("- 本期没有同步到失败 CI。\n" if not failed_ci else "")
            + "\n"
            "## 需要关注的 PR\n"
            + "\n".join(f"- #{pr.get('number')} {pr.get('title')}" for pr in prs[:5])
            + "\n\n## 失败 CI\n"
            + "\n".join(f"- {run.get('name')}: {run.get('conclusion')}" for run in failed_ci[:5])
            + "\n\n## 下周建议\n"
            "- 补充关键模块测试覆盖。\n- 对高风险 PR 提前拆分 Review。\n"
        )
        return {"report_markdown": markdown}
