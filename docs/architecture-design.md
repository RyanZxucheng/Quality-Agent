# 医学数据质量评估 Agent 架构设计

> **版本**: MVP (方案A - 轻量级)
> **日期**: 2026-04-08
> **框架**: Python + LLM
> **核心机制**: 工具检索证据 → LLM基于证据评分 → 代码决策

---

## 1. 项目概述

### 1.1 目标
构建一个自动化 Agent，用于评估医学问答（Question-Answer）数据的质量。通过**工具收集证据** + **LLM基于证据评分**的方式，实现高质量的质量评估。

### 1.2 输入输出

**输入**
- 格式: JSON/CSV/JSONL 文件
- 内容: 医学 QA 对（question + answer）
- 示例:
  ```json
  {
    "id": "qa_001",
    "question": "糖尿病患者可以服用二甲双胍吗？",
    "answer": "可以，二甲双胍是2型糖尿病的一线用药..."
  }
  ```

**输出**
- `cleaned_data/retained_qa.jsonl` - 保留的高质量数据
- `rejected/discarded_qa.jsonl` - 丢弃的数据及原因
- `reports/evaluation_report.json` - 详细评估报告（含证据和评分理由）

### 1.3 核心机制
1. **证据收集**: 并行调用多个工具（NER、术语验证、指南检查、维基百科）收集与QA相关的客观证据
2. **LLM评分**: 基于检索到的证据，由LLM进行主观质量评分
3. **代码决策**: 根据评分结果和阈值，由代码做出保留/丢弃的最终决策

### 1.4 核心约束
- 模型缺乏医学专业知识，必须通过工具调用获取外部知识作为评分依据
- 追求评估可解释性：每条评分都有工具证据支持
- 批量处理，自动丢弃低质量数据

---

## 2. 系统架构

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                          输入层 (Input Layer)                        │
│    JSON/CSV/JSONL Loader → QA Pair Parser                            │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    证据收集层 (Evidence Collection)                   │
│                                                                     │
│   并行调用工具检索相关证据:                                            │
│   ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐      │
│   │ EntityExtractor │ │ Terminology     │ │ Guideline       │      │
│   │ (医学实体识别)   │ │ Validator       │ │ Checker         │      │
│   │                 │ │ (术语标准化)     │ │ (指南符合性)     │      │
│   └─────────────────┘ └─────────────────┘ └─────────────────┘      │
│   ┌─────────────────┐                                               │
│   │ Wikipedia       │                                               │
│   │ Verifier        │                                               │
│   │ (概念验证)       │                                               │
│   └─────────────────┘                                               │
│                              │                                      │
│                              ▼                                      │
│                    Evidence Summary (文本摘要)                       │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     评分引擎层 (LLM Scoring)                         │
│                                                                     │
│   输入: QA + Evidence Summary                                       │
│                                                                     │
│   LLM 基于证据评分:                                                  │
│   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                  │
│   │ 完整性      │ │ 准确性       │ │ 专业性       │                  │
│   │ (0-30分)    │ │ (0-45分)     │ │ (0-25分)     │                  │
│   │             │ │             │ │             │                  │
│   │ • 信息充分  │ │ • 医学正确  │ │ • 表达专业  │                  │
│   │ • 实体覆盖  │ │ • 术语准确  │ │ • 免责声明  │                  │
│   └─────────────┘ └─────────────┘ └─────────────┘                  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      决策层 (Decision Layer)                         │
│                                                                     │
│   代码逻辑判断:                                                      │
│                                                                     │
│   IF 总分 ≥ 70 AND 准确性 ≥ 35                                      │
│      → 结论: RETAIN (保留)                                          │
│   ELSE                                                              │
│      → 结论: DISCARD (丢弃)                                         │
└────────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        输出层 (Output Layer)                         │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│   │  cleaned_data/   │  │  quality_report/ │  │   rejected/     │  │
│   │  (保留的数据集)   │  │  (评分+证据+理由) │  │ (丢弃的数据+原因)│  │
│   └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 工具设计

### 3.1 工具清单

| 工具名 | 功能 | 知识源 | 可靠性 | 覆盖维度 |
|--------|------|--------|--------|----------|
| `EntityExtractorTool` | 识别疾病、症状、药物、剂量等医学实体 | scispaCy 模型 | 中 | 完整性、专业性 |
| `TerminologyValidatorTool` | 验证术语是否符合标准医学术语 | ICD-10, SNOMED CT, UMLS | 极高 | 准确性、专业性 |
| `GuidelineCheckerTool` | 对比临床指南验证治疗方案 | NCCN/WHO 指南 | 极高 | 准确性 |
| `WikipediaVerifierTool` | 基础医学概念交叉验证 | Wikipedia API | 中 | 准确性（补充） |

### 3.2 工具使用策略

```python
# 工具优先级（可靠性排序）
TOOL_PRIORITY = [
    "TerminologyValidatorTool",  # Tier 1: 权威术语标准
    "GuidelineCheckerTool",      # Tier 1: 权威临床指南
    "EntityExtractorTool",       # Tier 2: 结构化解析
    "WikipediaVerifierTool",     # Tier 3: 补充验证
]

# 知识源权重（用于准确性评分计算）
SOURCE_WEIGHTS = {
    "terminology": 0.4,    # ICD-10/SNOMED
    "guideline": 0.4,      # 临床指南
    "wikipedia": 0.2,      # 维基百科
}
```

---

## 4. 质量评估维度

### 4.1 评分维度与权重

| 维度 | 权重 | 分值范围 | 评估内容 | LLM 评分依据 |
|------|------|----------|----------|--------------|
| **完整性** | 30% | 0-30 | 问题清晰、回答充分、信息完整 | 问题长度、回答长度、实体数量 |
| **准确性** | 45% | 0-45 | 医学知识正确、术语准确、符合指南 | **工具证据**: 术语库匹配、指南符合性、百科验证 |
| **专业性** | 25% | 0-25 | 术语规范、表达专业、有免责声明 | 术语规范性、结构化表达、免责声明 |
| **总计** | 100% | 0-100 | - | - |

**说明**: 所有维度由 **LLM 基于工具检索到的证据** 进行评分，而非纯规则计算。

### 4.2 维度评分细则

#### 完整性 (30分)

LLM 基于以下信息评分:
- 问题是否清晰完整（结合实体识别结果）
- 回答是否充分回应问题
- 信息是否完整（回答长度、结构）

#### 准确性 (45分) ⭐ 关键维度

LLM 基于**工具证据**评分:

| 证据来源 | 权重 | 说明 |
|----------|------|------|
| 术语验证 | 40% | ICD-10/SNOMED 匹配情况 |
| 指南检查 | 40% | 治疗方案是否符合临床指南 |
| 百科验证 | 20% | 基础医学概念是否可验证 |

**高准确性要求**: 此维度必须 ≥ 35 分才保留（见 5.1 决策规则）

#### 专业性 (25分)

LLM 基于以下内容评分:
- 术语使用规范性（参考术语验证结果）
- 表达是否专业（避免口语化）
- 是否包含医疗免责声明
- 回答结构是否清晰

---

## 5. 决策逻辑

### 5.1 保留/丢弃规则

**核心原则**: 代码基于 LLM 评分结果做决策，而非 LLM 直接输出结论

```python
def make_decision(scores: dict) -> tuple[str, str]:
    """
    根据 LLM 评分决定保留或丢弃

    规则:
    1. 总分必须 ≥ 70
    2. 准确性必须 ≥ 35 (关键维度，确保医学事实正确)
    3. 以上任一不满足则丢弃

    Returns:
        (conclusion, reason): 结论和原因
    """
    total = scores["completeness"] + scores["accuracy"] + scores["professionalism"]

    if total < 70:
        return "DISCARD", f"总分 {total} 低于阈值 70"

    if scores["accuracy"] < 35:
        return "DISCARD", f"准确性 {scores['accuracy']} 低于阈值 35"

    return "RETAIN", f"质量合格（总分 {total}, 准确性 {scores['accuracy']}）"
```

### 5.2 阈值调整建议

| 场景 | 总分阈值 | 准确性阈值 | 说明 |
|------|----------|------------|------|
| **保守模式** | 75 | 40 | 宁可漏掉不错过，适合对质量要求极高的场景 |
| **标准模式** | 70 | 35 | 平衡质量与数量（默认） |
| **宽松模式** | 65 | 30 | 保留更多数据用于人工复核 |

---

## 6. 输出格式

### 6.1 评估报告 (JSON)

```json
{
  "evaluation_summary": {
    "total_processed": 1000,
    "retained": 720,
    "discarded": 280,
    "retention_rate": "72%",
    "average_score": 78.5,
    "dimension_averages": {
      "completeness": 26.2,
      "accuracy": 35.8,
      "professionalism": 21.5
    }
  },
  "detailed_results": [
    {
      "id": "qa_001",
      "question": "糖尿病患者可以服用二甲双胍吗？",
      "answer": "可以，二甲双胍是2型糖尿病的一线用药...",
      "evidence_summary": "识别实体: 糖尿病、二甲双胍、2型糖尿病... 术语标准化率: 95%...",
      "scores": {
        "completeness": {"score": 28, "reason": "回答充分，包含用药机制和注意事项"},
        "accuracy": {"score": 43, "reason": "术语准确，符合NCCN指南推荐"},
        "professionalism": {"score": 23, "reason": "表达专业，有免责声明"},
        "total": 94
      },
      "conclusion": "RETAIN",
      "conclusion_reason": "质量优秀（总分 94, 准确性 43）"
    },
    {
      "id": "qa_002",
      "question": "感冒吃什么药好？",
      "answer": "吃点头孢就好了...",
      "evidence_summary": "识别实体: 感冒、头孢。术语: 头孢匹配。指南: 发现禁忌问题...",
      "scores": {
        "completeness": {"score": 18, "reason": "回答过于简短，缺乏详细说明"},
        "accuracy": {"score": 15, "reason": "抗生素滥用，普通感冒不应使用头孢"},
        "professionalism": {"score": 12, "reason": "表述口语化，缺少免责声明"},
        "total": 45
      },
      "conclusion": "DISCARD",
      "conclusion_reason": "准确性 15 低于阈值 35"
    }
  ]
}
```

### 6.2 输出文件结构

```
output/
├── cleaned_data/
│   ├── retained_qa.jsonl       # 保留的数据（JSON Lines格式）
│   └── retained_summary.csv    # 简要统计（ID, 总分, 结论）
├── rejected/
│   ├── discarded_qa.jsonl      # 丢弃的数据
│   └── discard_reasons.json    # 丢弃原因分类统计
└── reports/
    ├── evaluation_report.json  # 完整评估报告
    └── quality_distribution.png # 质量分布可视化（可选）
```

### 6.3 Issues 类型定义

| 问题类型 | 检测工具 | 示例 |
|----------|----------|------|
| 未知医学术语 | TerminologyValidator | "疾病X"不在ICD-10中 |
| 与指南冲突 | GuidelineChecker | 治疗方案与NCCN指南不符 |
| 实体缺失 | EntityExtractor | 诊断中未识别出疾病实体 |
| 术语不规范 | TerminologyValidator | 使用口语而非标准医学术语 |
| 事实存疑 | WikipediaVerifier | 基础医学事实与百科描述不符 |
| 语法错误 | LLM自检 | 错别字、病句 |

---

## 7. 知识源策略

### 7.1 知识源可靠性分级

| 来源 | 可靠性 | 出错概率 | 用途 |
|------|--------|----------|------|
| ICD-10 / SNOMED CT | ⭐⭐⭐⭐⭐ | 极低 | 术语标准化、编码验证 |
| 临床指南（NCCN/WHO） | ⭐⭐⭐⭐⭐ | 低 | 治疗方案验证 |
| PubMed 文献 | ⭐⭐⭐⭐ | 较低 | 事实验证 |
| UMLS | ⭐⭐⭐ | 低-中 | 术语映射 |
| 维基百科 | ⭐⭐ | 中 | 基础概念交叉验证（补充） |

### 7.2 维基百科使用策略

- **用途**: 仅用于基础概念验证和实体消歧
- **限制**: 不作为临床决策的唯一依据
- **权重**: 在准确性评分中权重较低 (20%)
- **场景示例**:
  - ✅ 验证"心脏有几个心房"这类基础解剖知识
  - ✅ 确认疾病的基本定义和分类
  - ❌ 判断具体治疗方案是否正确
  - ❌ 验证药物剂量是否安全

---

## 8. 扩展性设计

### 8.1 可插拔工具接口

```python
class MedicalValidationTool(Tool):
    """所有验证工具的基类"""

    name: str
    description: str
    reliability_tier: int  # 1-5，影响权重

    def validate(self, qa_pair: QAPair) -> ValidationResult:
        """返回验证结果和置信度"""
        pass
```

### 8.2 配置化维度权重

```yaml
# config/scoring_weights.yaml
dimensions:
  completeness:
    weight: 30
    sub_criteria:
      field_complete: 10
      entity_coverage: 10
      information_sufficient: 10

  accuracy:
    weight: 35
    sub_criteria:
      terminology_match: 15
      guideline_compliance: 15
      fact_verification: 5
    source_weights:
      terminology: 0.4
      guideline: 0.4
      wikipedia: 0.2

  professionalism:
    weight: 25
    sub_criteria:
      term_standardization: 15
      expression_professional: 10

  basic_quality:
    weight: 10
    sub_criteria:
      grammar_correct: 5
      readability: 5

thresholds:
  total_min: 70
  accuracy_min: 25
```

---

## 9. 性能与优化

### 9.1 批处理策略

- **逐条处理**: 避免内存溢出
- **断点续传**: 记录处理进度，支持中断后恢复
- **并行限制**: 控制并发请求数，避免 API 限流

### 9.2 缓存策略

- **术语缓存**: 已验证的术语结果本地缓存
- **指南缓存**: 指南内容定期更新而非实时查询
- **维基百科缓存**: API 结果缓存 7 天

---

## 10. 风险与限制

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 术语库覆盖不全 | 误判罕见病 | 结合维基百科补充，人工复核低置信度结果 |
| 指南更新滞后 | 过时治疗方案被误判为正确 | 定期更新指南库，标注指南版本 |
| API 调用失败 | 评估中断 | 实现重试机制和降级策略 |
| 维基百科错误 | 错误验证通过 | 降低维基百科权重，仅作交叉验证 |
| 评估耗时过长 | 大批量处理效率低 | 引入缓存、并行处理、增量评估 |

---

## 11. 后续迭代方向

1. **阶段 2**: 引入 ExpertRuleEngineTool，支持专家规则验证
2. **阶段 3**: 增加知识图谱（Neo4j），支持关系推理
3. **阶段 4**: 引入多 Agent 协作，专门 Agent 负责不同维度
4. **阶段 5**: 支持实时 API 服务，不仅限于批量处理

---

## 12. 附录

### 12.1 术语表

| 术语 | 说明 |
|------|------|
| ICD-10 | 国际疾病分类第10版 |
| SNOMED CT | 系统化医学命名法临床术语 |
| UMLS | 统一医学语言系统 |
| NCCN | 美国国家综合癌症网络 |
| NER | 命名实体识别 |

### 12.2 参考资源

- [ICD-10 官方编码](https://icd.who.int/browse10/2019/en)
- [SNOMED CT 浏览器](https://browser.ihtsdotools.org/)
- [UMLS API 文档](https://documentation.uts.nlm.nih.gov/)
- [scispaCy 医学 NER](https://allenai.github.io/scispacy/)
