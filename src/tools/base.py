"""
工具基类
"""
from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Dict[str, Any]
    error: str = ""


class BaseTool(ABC):
    """工具基类"""

    name: str = ""
    description: str = ""
    reliability_tier: int = 3  # 1-5，影响权重

    @abstractmethod
    def execute(self, question: str, answer: str) -> ToolResult:
        """执行工具"""
        pass

    def validate_input(self, question: str, answer: str) -> bool:
        """验证输入"""
        return bool(question and answer and len(question.strip()) > 0 and len(answer.strip()) > 0)
