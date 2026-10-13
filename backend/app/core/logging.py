"""系统日志配置模块.

本模块负责在应用程序启动初期初始化根日志记录器（Root Logger）的输出级别与日志格式。
"""

import logging


def configure_logging() -> None:
    """初始化全局日志配置.

    【Python 语法要点 - logging 标准库】:
    - `logging.basicConfig(...)`:
      设置根记录器的基本配置。若未提前配置，其他模块调用 `logging.getLogger(__name__)`
      时将使用默认的 WARNING 级别并输出到 stderr。
    - `format="%(asctime)s %(levelname)s [%(name)s] %(message)s"`:
      使用标准占位符模板定制格式化字符串：
      - `%(asctime)s`: 日志事件生成的人类可读时间戳。
      - `%(levelname)s`: 文本形式的日志级别（如 INFO, ERROR, WARNING）。
      - `%(name)s`: 记录器名称（通常为模块的 `__name__`）。
      - `%(message)s`: 实际打印的日志文本。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
