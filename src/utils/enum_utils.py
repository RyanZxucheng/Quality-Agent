"""
枚举工具模块
处理枚举与字符串之间的转换
"""
from enum import Enum
from typing import Type, TypeVar, Optional

E = TypeVar('E', bound=Enum)


def str_to_enum(enum_cls: Type[E], value: str, default: Optional[E] = None) -> E:
    """
    将字符串转换为枚举值

    Args:
        enum_cls: 枚举类
        value: 字符串值
        default: 转换失败时的默认值

    Returns:
        枚举值

    Raises:
        ValueError: 如果转换失败且未提供默认值
    """
    if not value:
        if default is not None:
            return default
        raise ValueError(f"Empty value cannot be converted to {enum_cls.__name__}")

    # 尝试直接匹配
    try:
        return enum_cls(value)
    except ValueError:
        pass

    # 尝试小写匹配
    try:
        return enum_cls(value.lower())
    except ValueError:
        pass

    # 尝试忽略大小写匹配
    for member in enum_cls:
        if str(member.value).lower() == value.lower():
            return member

    if default is not None:
        return default

    raise ValueError(f"Cannot convert '{value}' to {enum_cls.__name__}")


def enum_to_str(enum_value: E) -> str:
    """
    将枚举值转换为字符串

    Args:
        enum_value: 枚举值

    Returns:
        字符串表示
    """
    return enum_value.value


def is_valid_enum_value(enum_cls: Type[E], value: str) -> bool:
    """
    检查字符串是否为有效的枚举值

    Args:
        enum_cls: 枚举类
        value: 字符串值

    Returns:
        是否为有效值
    """
    try:
        str_to_enum(enum_cls, value)
        return True
    except ValueError:
        return False