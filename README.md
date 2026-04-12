# Medical QA Quality Agent

医学问答数据质量评估 Agent

## 项目简介

基于 **多轮自适应证据收集 + LLM 评分** 的自动化质量评估系统，评估医学问答（QA）数据的质量。

核心流程：
1. **基础工具并行收集**：NER、术语标准化、指南检查、维基百科验证
2. **Round 0 自检**：LLM 判断现有证据是否足够，识别信息缺口
3. **Round 1 内部检索**（按需）：BM25 + TF-IDF 混合召回 → RRF 融合 → 邻域扩展
4. **Round 2 自检复核**（按需）：结合内部检索结果再次判断
5. **Round 3 外部检索**（按需）：PubMed / Bing Search 等权威来源
6. **LLM 评分**：基于三层证据（BASE / INTERNAL / EXTERNAL）进行评分
7. **代码决策**：评分 + 证据不足标志共同决定保留或丢弃

## 核心特性

- **自检优先**：先用现有上下文判断，只有在确有缺口时才逐步触发检索
- **自适应检索深度**：按需触发内部 → 外部，避免无效调用
- **三层证据可信度**：EXTERNAL > INTERNAL > BASE，评分时优先信任高可靠来源
- **证据不足兜底**：全部轮次后仍不足时，标记"待人工复核"
- **多后端支持**：Anthropic Claude、vLLM 本地、OpenAI 兼容接口
- **可配置可观察**：所有阈值、工具、路径均通过 `config/` YAML 文件管理

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 首次使用 scispaCy 还需下载医学 NLP 模型：
> ```bash
> pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz
> ```

### 2. 配置 API Key

```bash
# Anthropic（默认）
export ANTHROPIC_API_KEY="sk-ant-..."

# 或 OpenAI
export OPENAI_API_KEY="sk-..."
```

### 3. 准备数据

将医学 QA 数据放入 `data/input/`，支持 JSON / JSONL / CSV 格式：

```json
[
  {
    "id": "qa_001",
    "question": "2型糖尿病患者可以服用二甲双胍吗？",
    "answer": "可以。二甲双胍是2型糖尿病的一线治疗药物..."
  }
]
```

### 4. 运行评估

```bash
# 最简运行（使用默认配置，仅基础工具 + 自检，不启用外部检索）
python -m src.main data/input/example_qa.json

# 指定输出目录
python -m src.main data/input/example_qa.json -o data/output

# 使用 OpenAI
python -m src.main data/input/example_qa.json \
    --llm-provider openai \
    --llm-model gpt-4o

# 使用本地 vLLM
python -m src.main data/input/example_qa.json \
    --llm-provider vllm \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --llm-base-url http://localhost:8000/v1

# 调整阈值
python -m src.main data/input/example_qa.json \
    --total-threshold 75 \
    --accuracy-threshold 40
```

### 5. 查看结果

```
data/output/
├── cleaned_data/
│   └── retained_qa.jsonl       # 保留的高质量数据（含多轮证据信息）
├── rejected/
│   └── discarded_qa.jsonl      # 丢弃数据及原因
├── reports/
│   ├── evaluation_report.json  # 完整报告（含每轮自检结果）
│   └── summary.txt             # 可读摘要（含多轮统计）
└── summary.json                # 统计摘要
```

---

## 启用内部检索（Round 1）

内部检索需要先准备知识库索引文件。

### 1. 准备索引文件

在 `data/index/` 目录下创建 `chunks.jsonl`，每行一个文档片段：

```jsonl
{"chunk_id": "chunk_001", "doc_id": "nccn_nsclc_2024", "content": "EGFR突变阳性非小细胞肺癌一线推荐奥希替尼80mg/天...", "chunk_index": 0, "metadata": {"source": "NCCN 2024", "section": "NSCLC"}}
{"chunk_id": "chunk_002", "doc_id": "nccn_nsclc_2024", "content": "对于EGFR外显子19缺失或21 L858R突变患者...", "chunk_index": 1, "metadata": {}}
```

### 2. 开启内部检索

编辑 `config/internal_search.yaml`：

```yaml
enabled: true          # 改为 true
index_dir: data/index  # 索引目录
bm25_top_k: 10
vector_top_k: 10
rerank_top_n: 3
coverage_threshold: 0.5
neighborhood_window: 1
```

---

## 启用外部检索（Round 3）

### PubMed（免费，无需 API Key）

编辑 `config/external_tools.yaml`：

```yaml
enabled: true   # 改为 true

tools:
  - name: pubmed
    enabled: true
    priority: 1
    query_template: "{missing_slot} clinical evidence"
    max_results: 2
```

### Bing Search（需要 Azure Key）

```yaml
enabled: true

tools:
  - name: bing_search
    enabled: true
    priority: 1
    api_key: "your-azure-key-here"
    endpoint: "https://api.bing.microsoft.com/v7.0/search"
    query_template: "{missing_slot} medical guideline"
    max_results: 2
```

---

## 配置文件说明

| 文件 | 说明 |
|------|------|
| `config/self_check.yaml` | 自检置信度阈值、prompt 路径 |
| `config/internal_search.yaml` | 内部检索参数（BM25/向量/重排/邻域） |
| `config/external_tools.yaml` | 外部工具列表、优先级、API 配置 |
| `config/prompts/self_check.md` | 自检系统 prompt（含 few-shot 示例） |

---

## CLI 参数一览

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--llm-provider` | 字符串 | `anthropic` / `openai` / `vllm` | `anthropic` |
| `--llm-model` | 字符串 | 模型名称 | `claude-3-5-sonnet-20241022` |
| `--llm-base-url` | 字符串 | 自定义 API 地址（vLLM 使用） | — |
| `--llm-api-key` | 字符串 | API 密钥（未指定时读环境变量） | — |
| `--llm-temperature` | 浮点数 | 生成温度 | `0.1` |
| `--llm-max-tokens` | 整数 | 最大 token 数 | `1500` |
| `--total-threshold` | 整数 | 总分阈值 | `70` |
| `--accuracy-threshold` | 整数 | 准确性阈值 | `35` |
| `--checkpoint-interval` | 整数 | 检查点间隔 | `100` |
| `-o`, `--output` | 字符串 | 输出目录 | `data/output` |

> 完整参数：`python -m src.main --help`

---

## 项目结构

```
Quality-Agent/
├── config/
│   ├── self_check.yaml          # 自检配置
│   ├── internal_search.yaml     # 内部检索配置
│   ├── external_tools.yaml      # 外部工具配置
│   └── prompts/
│       └── self_check.md        # 自检 prompt 模板
├── data/
│   ├── input/                   # 输入数据
│   ├── index/                   # 内部知识库索引（chunks.jsonl）
│   └── output/                  # 评估结果
├── docs/
│   ├── architecture-design.md
│   ├── InternalSearchExecutor.md
│   └── evidence-collection-redesign.md
└── src/
    ├── evidence/
    │   ├── collector.py         # 多轮证据收集编排
    │   ├── self_checker.py      # Round 0/2 自检模块
    │   └── internal_search.py  # Round 1 内部检索
    ├── tools/
    │   ├── entity_extractor.py
    │   ├── terminology_validator.py
    │   ├── wikipedia_verifier.py
    │   ├── guideline_checker.py
    │   └── external_search.py  # Round 3 外部检索
    ├── scoring/
    │   └── llm_engine.py        # LLM 三层证据评分
    ├── decision/
    │   └── engine.py            # 决策引擎（含证据不足判断）
    ├── report/
    │   └── generator.py         # 报告生成（含多轮证据链）
    └── models.py                # 数据模型
```

---

## 决策规则

| 条件 | 结论 |
|------|------|
| 总分 ≥ 70 **且** 准确性 ≥ 35 | **RETAIN**（保留） |
| 总分 < 70 或 准确性 < 35 | **DISCARD**（丢弃） |
| 存在严重医学错误（指南违禁） | **DISCARD** |
| 全部轮次后证据仍不足 | **DISCARD**（待人工复核） |

---

## 技术栈

- Python 3.10+
- Anthropic / OpenAI SDK
- scispaCy（医学 NER）
- rank-bm25（关键词检索）
- scikit-learn（TF-IDF 向量检索）
- PubMed E-utilities API（外部医学文献检索）
- vLLM / OpenAI 兼容接口（本地模型）
