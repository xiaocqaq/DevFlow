"""智能体抽象基类模块.

本模块定义了 DevFlow 智能体系统中所有 Agent（如 IssueAgent、PRReviewAgent、
PlannerAgent 等）必须遵循的接口规范与通用错误处理骨架。
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """智能体抽象基类 (Abstract Base Class).

    【Python 语法要点 - 面向对象与抽象基类 (ABC)】:
    1. 继承自 `abc.ABC`:
       - 标记此类为抽象基类，用于制定契约（Contract），规范派生子类的行为。
       - 只要类中存在至少一个带有 `@abstractmethod` 装饰器的方法，该基类就不能被直接实例化。
    2. `@abstractmethod` 装饰器:
       - 强制要求所有继承 `BaseAgent` 的具体子类必须重写并实现该方法；
       - 若子类未实现 `@abstractmethod` 装饰的方法，在尝试实例化该子类时，Python 解释器
         会在运行时抛出 `TypeError: Can't instantiate abstract class ... with abstract method run`。
    3. 异步协程方法 `async def`:
       - 契约定义为异步函数，表明 Agent 的内部核心逻辑依赖非阻塞 I/O（如请求 LLM 接口、调用向量库、访问 GitHub API）。
    4. 类变量类型声明 (Type Annotations):
       - `name: str`: 智能体唯一标识名称。
       - `description: str`: 智能体功能说明描述。
       - `available_tools: list[str]`: 该智能体具备或允许调用的工具名称列表。
    """

    name: str
    description: str
    available_tools: list[str]

    @abstractmethod
    async def run(self, input_data: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """执行智能体的核心任务逻辑.

        Args:
            input_data (dict[str, Any]): 任务入参数据（如 issue_id、pr_number、用户提问等）。
            context (dict[str, Any]): 上下文依赖（如 repo_id、数据库连接、已加载的短期记忆等）。

        Returns:
            dict[str, Any]: 结构化任务执行产出。

        Raises:
            NotImplementedError: 抽象方法默认抛出，由子类负责具体实现。
        """
        raise NotImplementedError

    def validate_output(self, output: dict[str, Any]) -> dict[str, Any]:
        """校验并过滤智能体的产出数据（钩子方法 / Hook Method）.

        子类可选择性覆盖（Override）此方法以执行模式校验（Schema Validation）或安全合规过滤。
        基类默认提供直通（Pass-through）实现。

        Args:
            output (dict[str, Any]): 原始产出数据字典。

        Returns:
            dict[str, Any]: 校验/清洗后的产出字典。
        """
        return output

    def handle_error(self, error: Exception, context: dict[str, Any]) -> dict[str, Any]:
        """统一错误降级处理方法.

        当 Agent 在执行或工具调用中发生异常时，统一组装标准的错误包络结构返回，
        避免异常向上击穿导致工作流中断。

        【Python 语法要点 - 字典操作与条件表达式】:
        - `context.get("repo_id")`: 安全访问字典键，避免 KeyError。
        - 三元条件表达式 `x if condition else y`。

        Args:
            error (Exception): 捕获到的异常对象。
            context (dict[str, Any]): 运行时上下文。

        Returns:
            dict[str, Any]: 包含错误原因与上下文元数据的字典。
        """
        return {
            "error": str(error),
            "agent": self.name,
            "context": {"repo_id": str(context.get("repo_id")) if context.get("repo_id") else None},
        }
