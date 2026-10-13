"""安全与数据加密模块.

本模块基于对称加密算法（Fernet）为敏感配置（如用户的 GitHub Token）
提供存储加密（Encryption at Rest）与安全解密支持。
"""

from cryptography.fernet import Fernet, InvalidToken


def _derive_demo_key(secret: str) -> bytes:
    """根据输入的明文密钥字符串派生符合 Fernet 要求的 32 字节密钥.

    【Python 语法要点 - 字节串与加密哈希】:
    - 局部导入（Local Import）：将 `base64` 与 `hashlib` 放在函数内导入，减小模块初始加载开销。
    - `secret.encode("utf-8")`: 将 Python 字符串（str）编码为 UTF-8 字节串（bytes）。
    - `hashlib.sha256(...).digest()`: 对字节串做 SHA-256 哈希运算，返回固定 32 字节（256 位）的原始二进制摘要。
    - `base64.urlsafe_b64encode(...)`: 将 32 字节二进制摘要转换为 URL 安全的 Base64 编码字节串，满足 Fernet 构造函数的硬性格式规范。

    Args:
        secret (str): 原始明文密钥或密码种子。

    Returns:
        bytes: 经 URL-safe Base64 编码的 32 字节派生密钥。
    """
    import base64
    import hashlib

    return base64.urlsafe_b64encode(hashlib.sha256(secret.encode("utf-8")).digest())


def encrypt_token(token: str, secret: str) -> str:
    """使用对称加密算法加密敏感 Token.

    【Python 语法要点 - 对称加密】:
    - `Fernet(key).encrypt(data)`:
      Fernet 使用 AES-128-CBC 模式加密，配合 HMAC-SHA256 提供消息认证（Authenticated Cryptography），
      内置时间戳并包含防篡改校验。
    - 链式调用：将 token 编码为 bytes -> 调用 Fernet 加密 -> 将密文字节解码回可读 str 字符串保存到数据库。

    Args:
        token (str): 待加密的明文字符串（如 GitHub Personal Access Token）。
        secret (str): 加密主密钥。

    Returns:
        str: 加密后的密文字符串（Base64 表现形式）。
    """
    return Fernet(_derive_demo_key(secret)).encrypt(token.encode("utf-8")).decode("utf-8")


def decrypt_token(encrypted: str, secret: str) -> str | None:
    """解密存储的敏感密文.

    【Python 语法要点 - 异常捕获与可选返回值】:
    - `str | None`: 返回类型提示，表示若解密成功返回明文字符串，若密钥错误或密文被篡改则返回 None。
    - `try ... except InvalidToken`:
      Fernet 在密钥不匹配、时间戳失效或密文被篡改时会抛出 `InvalidToken` 异常。
      捕获该异常避免程序崩溃，优雅降级返回 None。

    Args:
        encrypted (str): 已加密的密文字符串。
        secret (str): 解密主密钥。

    Returns:
        str | None: 解密后的明文字符串，若解密失败（如凭证无效）则返回 None。
    """
    try:
        return Fernet(_derive_demo_key(secret)).decrypt(encrypted.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        return None
