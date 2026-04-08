"""
术语验证工具
验证医学术语是否符合标准（ICD-10, SNOMED CT）
MVP 版本使用本地术语表，生产版本可对接 UMLS API
"""
import logging
import json
import re
import os
from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TerminologyValidatorTool(BaseTool):
    """
    医学术语验证工具
    验证术语是否符合 ICD-10、SNOMED CT 等标准
    """

    name = "terminology_validator"
    description = "验证医学术语的标准性（ICD-10, SNOMED CT）"
    reliability_tier = 5  # 最高可靠性

    def __init__(self, terminology_dir: Optional[str] = None):
        """
        Args:
            terminology_dir: 术语库目录路径
        """
        self.terminology_dir = terminology_dir or "data/terminology"
        self._icd10_codes: Dict[str, str] = {}  # code -> description
        self._snomed_terms: Set[str] = set()
        self._common_medical_terms: Set[str] = set()
        self._load_terminologies()

    def _load_terminologies(self):
        """加载术语库"""
        # 加载 ICD-10
        icd10_path = Path(self.terminology_dir) / "icd10_simple.json"
        if icd10_path.exists():
            with open(icd10_path, "r", encoding="utf-8") as f:
                self._icd10_codes = json.load(f)
            logger.info(f"Loaded {len(self._icd10_codes)} ICD-10 codes")
        else:
            logger.warning(f"ICD-10 file not found at {icd10_path}, using minimal set")
            self._icd10_codes = self._get_minimal_icd10()

        # 加载 SNOMED CT 术语
        snomed_path = Path(self.terminology_dir) / "snomed_simple.txt"
        if snomed_path.exists():
            with open(snomed_path, "r", encoding="utf-8") as f:
                self._snomed_terms = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(self._snomed_terms)} SNOMED terms")
        else:
            logger.warning(f"SNOMED file not found at {snomed_path}, using minimal set")
            self._snomed_terms = self._get_minimal_snomed()

        # 加载常用医学术语
        self._common_medical_terms = self._get_common_medical_terms()

    def _get_minimal_icd10(self) -> Dict[str, str]:
        """最小 ICD-10 集合（用于 MVP）"""
        return {
            "E11": "Type 2 diabetes mellitus",
            "I10": "Essential hypertension",
            "J06": "Acute upper respiratory infections",
            "K29": "Gastritis and duodenitis",
            "M79": "Other and unspecified soft tissue disorders",
            "N39": "Other disorders of urinary system",
            "F32": "Depressive episode",
            "G43": "Migraine",
            "H10": "Conjunctivitis",
            "L20": "Atopic dermatitis",
        }

    def _get_minimal_snomed(self) -> Set[str]:
        """最小 SNOMED 集合（用于 MVP）"""
        return {
            "diabetes mellitus", "hypertension", "pneumonia", "asthma",
            "migraine", "depression", "anxiety", "arthritis",
            "metformin", "insulin", "aspirin", "ibuprofen",
            "headache", "fever", "cough", "nausea",
            "heart", "lung", "liver", "kidney", "brain",
        }

    def _get_common_medical_terms(self) -> Set[str]:
        """常用医学术语集合"""
        return {
            # 疾病
            "diabetes", "hypertension", "hyperlipidemia", "pneumonia",
            "bronchitis", "asthma", "tuberculosis", "hepatitis",
            "cirrhosis", "nephritis", "anemia", "leukemia",
            "melanoma", "carcinoma", "sarcoma", "lymphoma",

            # 药物
            "metformin", "insulin", "glibenclamide", "aspirin",
            "acetaminophen", "ibuprofen", "amoxicillin", "cephalexin",
            "atorvastatin", "simvastatin", "amlodipine", "losartan",

            # 症状
            "headache", "dizziness", "nausea", "vomiting",
            "diarrhea", "constipation", "fever", "chills",
            "fatigue", "weakness", "pain", "swelling",

            # 解剖
            "heart", "lung", "liver", "kidney", "spleen",
            "pancreas", "stomach", "intestine", "brain", "spine",
        }

    def execute(self, question: str, answer: str) -> ToolResult:
        """执行术语验证"""
        if not self.validate_input(question, answer):
            return ToolResult(
                success=False,
                data={},
                error="Invalid input"
            )

        try:
            # 提取候选术语（这里简化处理，实际应使用 NER 结果）
            text = f"{question} {answer}".lower()
            words = self._extract_medical_terms(text)

            validation_results = []
            for word in words:
                result = self._validate_term(word)
                validation_results.append(result)

            # 计算统计
            total = len(validation_results)
            valid_icd10 = sum(1 for r in validation_results if r["in_icd10"])
            valid_snomed = sum(1 for r in validation_results if r["in_snomed"])
            valid_common = sum(1 for r in validation_results if r["in_common_terms"])

            # 标准化率 = 至少在任一标准库中找到的术语比例
            standardized = sum(1 for r in validation_results
                             if r["in_icd10"] or r["in_snomed"] or r["in_common_terms"])

            standardization_rate = standardized / total if total > 0 else 0

            return ToolResult(
                success=True,
                data={
                    "validation_results": validation_results,
                    "statistics": {
                        "total_terms": total,
                        "in_icd10": valid_icd10,
                        "in_snomed": valid_snomed,
                        "in_common_terms": valid_common,
                        "standardized": standardized,
                        "standardization_rate": standardization_rate,
                    },
                    "unstandardized_terms": [
                        r["term"] for r in validation_results
                        if not (r["in_icd10"] or r["in_snomed"] or r["in_common_terms"])
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Terminology validation failed: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e)
            )

    def _extract_medical_terms(self, text: str) -> List[str]:
        """提取候选医学术语（简化版）"""
        # 清理文本
        text = re.sub(r'[^\w\s]', ' ', text)

        # 提取单词和词组
        words = set()

        # 单字词
        single_words = text.split()
        words.update(w for w in single_words if len(w) >= 4)

        # 简单的双词组合
        tokens = text.split()
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            words.add(bigram)

        return list(words)[:20]  # 限制数量

    def _validate_term(self, term: str) -> Dict[str, Any]:
        """验证单个术语"""
        term_lower = term.lower()
        term_upper = term.upper()

        # 检查 ICD-10（代码或描述）
        in_icd10 = (
            term_upper in self._icd10_codes  # 代码匹配
            or any(term_lower in desc.lower() for desc in self._icd10_codes.values())  # 描述匹配
        )

        # 检查 SNOMED
        in_snomed = any(term_lower in concept.lower() for concept in self._snomed_terms)

        # 检查常用术语
        in_common = term_lower in self._common_medical_terms

        return {
            "term": term,
            "in_icd10": in_icd10,
            "in_snomed": in_snomed,
            "in_common_terms": in_common,
            "is_standardized": in_icd10 or in_snomed or in_common,
        }

    def validate_icd10_code(self, code: str) -> bool:
        """验证 ICD-10 代码（便捷方法）"""
        return code.upper() in self._icd10_codes

    def get_icd10_description(self, code: str) -> Optional[str]:
        """获取 ICD-10 代码描述（便捷方法）"""
        return self._icd10_codes.get(code.upper())
