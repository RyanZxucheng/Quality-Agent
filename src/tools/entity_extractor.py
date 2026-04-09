"""
医学实体识别工具
使用 scispaCy 识别医学实体
"""
import logging
from typing import List, Dict, Any, Optional
import spacy

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class EntityExtractorTool(BaseTool):
    """
    医学实体识别工具
    使用 scispaCy 的 en_core_sci_sm 模型
    """

    name = "entity_extractor"
    description = "识别医学文本中的实体（疾病、药物、症状等）"
    reliability_tier = 3

    # 实体类型映射（简化版）
    ENTITY_TYPE_MAPPING = {
        "DISEASE": ["Disease", "Disorder", "Syndrome", "Condition"],
        "DRUG": ["Drug", "Chemical", "Substance", "Medication"],
        "SYMPTOM": ["Sign", "Symptom", "Finding"],
        "ANATOMY": ["Anatomy", "Organ", "Tissue"],
        "PROCEDURE": ["Procedure", "Treatment", "Therapy", "Surgery"],
    }

    def __init__(self, model_name: str = ""):
        from src.config import get_config
        self.model_name = model_name or get_config().spacy_model
        self._nlp: Optional[Any] = None
        self._load_model()

    def _load_model(self):
        """加载 scispaCy 模型"""
        try:
            self._nlp = spacy.load(self.model_name)
            logger.info(f"Loaded scispaCy model: {self.model_name}")
        except OSError:
            logger.error(
                f"Model {self.model_name} not found. "
                "Please install it: pip install {model_url}"
            )
            raise

    def execute(self, question: str, answer: str) -> ToolResult:
        """执行实体识别"""
        if not self.validate_input(question, answer):
            return ToolResult(
                success=False,
                data={},
                error="Invalid input: question and answer cannot be empty"
            )

        try:
            # 合并 question 和 answer 进行识别
            combined_text = f"{question} {answer}"
            doc = self._nlp(combined_text)

            entities = []
            for ent in doc.ents:
                entity_type = self._normalize_entity_type(ent.label_)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "normalized_type": entity_type,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

            # 按类型分组
            entities_by_type = self._group_by_type(entities)

            return ToolResult(
                success=True,
                data={
                    "entities": entities,
                    "entities_by_type": entities_by_type,
                    "entity_count": len(entities),
                    "unique_entities": list(set(e["text"] for e in entities)),
                }
            )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e)
            )

    def _normalize_entity_type(self, label: str) -> str:
        """将 scispaCy 的实体类型归一化为标准类型"""
        for standard_type, scispacy_types in self.ENTITY_TYPE_MAPPING.items():
            if label in scispacy_types:
                return standard_type
        return "OTHER"

    def _group_by_type(self, entities: List[Dict]) -> Dict[str, List[str]]:
        """按类型分组实体"""
        grouped: Dict[str, List[str]] = {}
        for ent in entities:
            entity_type = ent["normalized_type"]
            if entity_type not in grouped:
                grouped[entity_type] = []
            if ent["text"] not in grouped[entity_type]:
                grouped[entity_type].append(ent["text"])
        return grouped

    def get_disease_entities(self, text: str) -> List[str]:
        """获取疾病实体（便捷方法）"""
        result = self.execute("", text)
        if result.success:
            return result.data.get("entities_by_type", {}).get("DISEASE", [])
        return []

    def get_drug_entities(self, text: str) -> List[str]:
        """获取药物实体（便捷方法）"""
        result = self.execute("", text)
        if result.success:
            return result.data.get("entities_by_type", {}).get("DRUG", [])
        return []
