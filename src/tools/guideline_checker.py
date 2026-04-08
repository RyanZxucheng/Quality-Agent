"""
临床指南检查工具
验证治疗方案是否符合临床指南
MVP 版本使用简化的规则匹配
"""
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class GuidelineCheckerTool(BaseTool):
    """
    临床指南检查工具
    验证 Q&A 中的诊疗建议是否符合临床指南
    """

    name = "guideline_checker"
    description = "验证诊疗建议是否符合临床指南"
    reliability_tier = 5  # 最高可靠性

    def __init__(self, guidelines_dir: Optional[str] = None):
        """
        Args:
            guidelines_dir: 指南文件目录
        """
        self.guidelines_dir = guidelines_dir or "data/guidelines"
        self._guidelines: Dict[str, Any] = {}
        self._treatment_rules: List[Dict] = []
        self._load_guidelines()

    def _load_guidelines(self):
        """加载临床指南"""
        guidelines_path = Path(self.guidelines_dir) / "simple_guidelines.json"

        if guidelines_path.exists():
            with open(guidelines_path, "r", encoding="utf-8") as f:
                self._guidelines = json.load(f)
            logger.info(f"Loaded guidelines from {guidelines_path}")
        else:
            logger.warning(f"Guidelines file not found, using minimal rules")
            self._guidelines = self._get_minimal_guidelines()

        # 构建治疗规则索引
        self._build_treatment_rules()

    def _get_minimal_guidelines(self) -> Dict:
        """最小指南集合（用于 MVP）"""
        return {
            "diabetes_type2": {
                "condition": "type 2 diabetes",
                "first_line": ["metformin", "lifestyle modification"],
                "contraindicated": ["insulin"],  # 不作为一线
                "keywords": ["diabetes", "type 2", "T2DM"]
            },
            "hypertension": {
                "condition": "essential hypertension",
                "first_line": ["amlodipine", "losartan", "lifestyle modification"],
                "contraindicated": [],
                "keywords": ["hypertension", "high blood pressure"]
            },
            "pneumonia_community": {
                "condition": "community acquired pneumonia",
                "first_line": ["amoxicillin", "azithromycin", "respiratory fluoroquinolone"],
                "contraindicated": [],
                "keywords": ["pneumonia", "CAP"]
            },
            "common_cold": {
                "condition": "common cold",
                "first_line": ["supportive care", "rest", "hydration"],
                "contraindicated": ["antibiotics"],  # 病毒感染禁用抗生素
                "keywords": ["cold", "upper respiratory infection", "viral"]
            },
        }

    def _build_treatment_rules(self):
        """构建治疗规则索引"""
        for condition_key, guideline in self._guidelines.items():
            self._treatment_rules.append({
                "condition_key": condition_key,
                "condition": guideline["condition"],
                "first_line": guideline.get("first_line", []),
                "contraindicated": guideline.get("contraindicated", []),
                "keywords": guideline.get("keywords", [])
            })

    def execute(self, question: str, answer: str) -> ToolResult:
        """执行指南检查"""
        if not self.validate_input(question, answer):
            return ToolResult(
                success=False,
                data={},
                error="Invalid input"
            )

        try:
            # 识别可能的疾病
            conditions = self._identify_conditions(question, answer)

            check_results = []
            for condition in conditions:
                result = self._check_condition_treatment(condition, question, answer)
                check_results.append(result)

            # 计算整体合规性
            if check_results:
                compliant_count = sum(1 for r in check_results if r["is_compliant"])
                compliance_rate = compliant_count / len(check_results)
            else:
                compliance_rate = 1.0  # 无匹配疾病，默认为合规

            return ToolResult(
                success=True,
                data={
                    "check_results": check_results,
                    "conditions_identified": len(conditions),
                    "compliance_rate": compliance_rate,
                    "issues": [
                        r for r in check_results
                        if not r["is_compliant"] and r.get("issues")
                    ]
                }
            )

        except Exception as e:
            logger.error(f"Guideline check failed: {e}")
            return ToolResult(
                success=False,
                data={},
                error=str(e)
            )

    def _identify_conditions(self, question: str, answer: str) -> List[str]:
        """识别文本中提到的疾病"""
        text = f"{question} {answer}".lower()
        conditions = []

        for rule in self._treatment_rules:
            for keyword in rule["keywords"]:
                if keyword.lower() in text:
                    conditions.append(rule["condition_key"])
                    break

        return list(set(conditions))  # 去重

    def _check_condition_treatment(self, condition_key: str, question: str, answer: str) -> Dict[str, Any]:
        """检查特定疾病的治疗方案"""
        rule = next(
            (r for r in self._treatment_rules if r["condition_key"] == condition_key),
            None
        )

        if not rule:
            return {"condition": condition_key, "is_compliant": True}

        answer_lower = answer.lower()
        issues = []

        # 检查是否推荐了一线治疗方案
        has_first_line = any(
            treatment.lower() in answer_lower
            for treatment in rule["first_line"]
        )

        # 检查是否推荐了禁忌药物
        contraindicated_found = []
        for drug in rule["contraindicated"]:
            if drug.lower() in answer_lower:
                contraindicated_found.append(drug)

        # 判断合规性
        is_compliant = True

        if contraindicated_found:
            is_compliant = False
            issues.append({
                "type": "contraindicated_treatment",
                "severity": "high",
                "description": f"推荐了禁忌药物: {', '.join(contraindicated_found)}",
                "condition": rule["condition"]
            })

        # 如果没有推荐一线方案且推荐了其他药物，可能存在问题
        # 简化：仅作为警告
        if not has_first_line and ("medication" in answer_lower or "drug" in answer_lower):
            issues.append({
                "type": "non_first_line_treatment",
                "severity": "medium",
                "description": f"治疗方案可能不符合 {rule['condition']} 的一线推荐",
                "expected": rule["first_line"]
            })

        return {
            "condition": rule["condition"],
            "condition_key": condition_key,
            "is_compliant": is_compliant and len([i for i in issues if i["severity"] == "high"]) == 0,
            "has_first_line_treatment": has_first_line,
            "contraindicated_found": contraindicated_found,
            "issues": issues
        }

    def get_guideline_for_condition(self, condition: str) -> Optional[Dict]:
        """获取特定疾病的指南（便捷方法）"""
        return self._guidelines.get(condition.lower().replace(" ", "_"))
