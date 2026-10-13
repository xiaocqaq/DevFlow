"""数据库会话与连接引擎模块.

本模块负责建立与 PostgreSQL 的连接池引擎，配置 SQLAlchemy ORM 会话工厂，
并提供供 FastAPI 依赖注入系统（Dependency Injection）使用的数据库会话生成器。
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    """SQLAlchemy 2.0 声明式模型基类.

    【Python 语法要点】:
    - 继承自 `DeclarativeBase`（SQLAlchemy 2.0+ 推荐语法，取代旧版的 `declarative_base()` 工厂函数）。
    - 所有的 ORM 数据模型类（如 User、Repository 等）均继承此类。
    - 类内部自动维护 `metadata`（元数据对象），用于记录所有关联表的表结构定义，
      供后续 `Base.metadata.create_all(bind=engine)` 自动建表或 Alembic 迁移脚本使用。
    """

    pass


# 数据库引擎配置
# 【Python / SQLAlchemy 语法要点】:
# - pool_pre_ping=True: 连接池健康检查（心跳机制），每次从连接池借出连接前执行一次类似 'SELECT 1' 的探测，
#   若发现底层数据库连接已断开（如因数据库重启或超时中断），会自动丢弃旧连接并重连，防止出现 500 报错。
engine = create_engine(settings.database_url, pool_pre_ping=True)

# 会话工厂配置
# 【SQLAlchemy 语法要点】:
# - autoflush=False: 禁用自动刷新。防止在执行查询前隐式将未提交的修改推到数据库，使事务控制更显式、可预测。
# - autocommit=False: 禁用自动提交。遵循现代 ORM 事务规范，要求开发者显式调用 session.commit() 进行事务提交。
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话的生成器函数（FastAPI 依赖项）.

    【Python 语法要点 - 生成器上下文 (Generator Context)】:
    - 返回类型注解 `Generator[Session, None, None]`:
      依次代表 `Generator[YieldType, SendType, ReturnType]`。
      此处通过 `yield` 返回一个 `Session` 实例，无外部传入值，无最终 `return` 返回值。
    - FastAPI 依赖注入 (`Depends(get_db)`):
      FastAPI 会在接收到 HTTP 请求时执行该函数至 `yield db`，将 `db` 注入到路由处理函数中；
      请求处理完毕（或发生异常）后，执行流程会回到 `finally` 块中执行 `db.close()`，
      确保即使发生网络异常或未捕获错误，数据库连接也能被及时释放回连接池，防止连接泄露。

    Yields:
        Session: 活跃的 SQLAlchemy 数据库会话对象。

    Examples:
        >>> @router.get("/items")
        >>> def read_items(db: Session = Depends(get_db)):
        >>>     return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
