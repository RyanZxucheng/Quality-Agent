"""
JSON 工具模块
包含LLM JSON解析、安全JSON加载等通用功能
"""
import json
import re
from typing import Dict, Any, Optional

# 预编译正则表达式用于JSON解析（性能优化）
_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL)


def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    从LLM输出中提取JSON

    Args:
        text: LLM输出的文本

    Returns:
        解析后的JSON字典

    Raises:
        ValueError: 如果无法解析JSON
    """
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试从代码块中提取
    json_match = _JSON_BLOCK_PATTERN.search(text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试从文本中定位第一个有效 JSON 对象
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, _ = decoder.raw_decode(text[pos:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        pos = text.find("{", pos + 1)

    raise ValueError(f"无法解析LLM输出: {text[:200]}...")


def safe_json_loads(text: str, default: Optional[Dict] = None) -> Dict[str, Any]:
    """
    安全地解析JSON，失败时返回默认值

    Args:
        text: JSON字符串
        default: 解析失败时返回的默认值

    Returns:
        解析后的JSON字典或默认值
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default or {}


def format_json_for_output(data: Dict, indent: int = 2) -> str:
    """
    格式化JSON用于输出，确保中文字符正常显示

    Args:
        data: 要格式化的数据
        indent: 缩进空格数

    Returns:
        格式化后的JSON字符串
    """
    return json.dumps(data, ensure_ascii=False, indent=indent)