"""
维基百科验证工具
用于基础医学概念的交叉验证
"""
import logging
import re
import requests
from typing import List, Dict, Any
import time

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class WikipediaVerifierTool(BaseTool):
    """
    维基百科验证工具
    用于验证基础医学概念，仅作为补充验证手段
    """

    name = "wikipedia_verifier"
    description = "使用维基百科验证基础医学概念"
    reliability_tier = 2  # 可靠性较低，仅作交叉验证

    BASE_URL = "https://en.wikipedia.org/api/rest_v1"
    SEARCH_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, rate_limit: float = 1.0):
        """
        Args:
            rate_limit: API 调用间隔（秒），避免请求过快
        """
        self.rate_limit = rate_limit
        self._last_request_time = 0

    def _respect_rate_limit(self):
        """遵守速率限制"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def execute(self, question: str, answer: str) -> ToolResult:
        """执行维基百科验证"""
        if not self.validate_input(question, answer):
            return ToolResult(
                success=False,
                data={},
                error="Invalid input"
            )

        try:
            # 提取关键实体进行验证
            entities_to_verify = self._extract_key_entities(question, answer)

            verification_results = []
            for entity in entities_to_verify:
                result = self._verify_entity(entity)
                verification_results.append(result)

            # 计算整体置信度
            if verification_results:
                avg_confidence = sum(r["confidence"] for r in verification_results) / len(verification_results)
            else:
                avg_confidence = 0.5  # 中性值

            return ToolResult(
                success=True,
                data={
                    "verification_results": verification_results,
                    "average_confidence": avg_confidence,
                    "entities_checked": len(verification_results),
                    "entities_found": sum(1 for r in verification_results if r["found"]),
                }
            )

        except Exception as e:
            logger.error(f"Wikipedia verification failed: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e)
            )

    def _extract_key_entities(self, question: str, answer: str) -> List[str]:
        """提取需要验证的关键实体（简化版）"""
        # 合并文本
        text = f"{question} {answer}"

        # 匹配首字母大写的单词组合（潜在的专业术语）
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

        # 去重并过滤常见词
        stop_words = {"The", "A", "An", "This", "That", "These", "Those"}
        entities = [m for m in set(matches) if m not in stop_words and len(m) > 3]

        # 限制数量，避免过多API调用
        return entities[:5]

    def _verify_entity(self, entity: str) -> Dict[str, Any]:
        """验证单个实体"""
        self._respect_rate_limit()

        try:
            # 使用 Wikipedia REST API 检查页面是否存在
            url = f"{self.BASE_URL}/page/summary/{entity.replace(' ', '_')}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    "entity": entity,
                    "found": True,
                    "title": data.get("title", ""),
                    "description": data.get("description", ""),
                    "extract": data.get("extract", "")[:200],  # 截断
                    "confidence": 0.8 if "medical" in data.get("description", "").lower() else 0.6
                }
            else:
                return {
                    "entity": entity,
                    "found": False,
                    "confidence": 0.0
                }

        except requests.RequestException as e:
            logger.warning(f"Failed to verify entity '{entity}': {e}")
            return {
                "entity": entity,
                "found": False,
                "error": str(e),
                "confidence": 0.0
            }

    def verify_medical_term(self, term: str) -> Dict[str, Any]:
        """验证医学术语（便捷方法）"""
        self._respect_rate_limit()

        try:
            # 搜索医学相关内容
            params = {
                "action": "query",
                "list": "search",
                "srsearch": f"{term} medical",
                "format": "json",
                "srlimit": 3
            }

            response = requests.get(self.SEARCH_URL, params=params, timeout=10)
            data = response.json()

            search_results = data.get("query", {}).get("search", [])

            if search_results:
                best_match = search_results[0]
                relevance_score = best_match.get("score", 0)

                return {
                    "term": term,
                    "found": True,
                    "matches": [r["title"] for r in search_results],
                    "best_match": best_match["title"],
                    "relevance": min(relevance_score / 1000, 1.0),  # 归一化
                }
            else:
                return {
                    "term": term,
                    "found": False,
                    "matches": []
                }

        except Exception as e:
            logger.error(f"Medical term verification failed: {e}")
            return {
                "term": term,
                "found": False,
                "error": str(e)
            }
