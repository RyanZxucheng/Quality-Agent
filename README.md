# Medical QA Quality Agent

医学问答数据质量评估 Agent

## 项目简介

基于 **工具检索证据 + LLM 评分** 的自动化质量评估系统，用于评估医学问答（Question-Answer）数据的质量。

核心流程：
1. **工具收集证据**：并行调用 NER、术语验证、指南检查、维基百科等工具
2. **LLM 基于证据评分**：使用 LLM（支持 Anthropic/vLLM/OpenAI）基于证据进行评分
3. **代码决策**：根据评分结果和阈值决定保留或丢弃

## 核心特性

- **证据驱动的评分**：工具检索客观证据，LLM 基于证据主观评分
- **多后端支持**：支持 Anthropic Claude、vLLM 本地部署、OpenAI 兼容接口
- **三维度评估**：完整性(30)、准确性(45)、专业性(25)，总分100
- **批量处理**：自动处理 JSON/CSV/JSONL 文件，输出清洗后的数据集

## 项目结构

```
medical-qa-quality-agent/
├── docs/
│   └── architecture-design.md     # 架构设计文档
├── src/                           # 源代码
│   ├── tools/                     # 证据收集工具（NER/术语/指南/百科等）
│   ├── evidence/                  # 证据收集编排
│   ├── scoring/                   # 评分引擎（LLM 评分等）
│   ├── decision/                  # 决策引擎（保留/丢弃）
│   ├── report/                    # 报告生成
│   └── utils/                     # 通用工具（JSON/文件/枚举等）
├── config/                        # 配置文件
├── data/
│   ├── input/                     # 输入数据
│   └── output/
│       ├── cleaned_data/          # 保留的数据
│       ├── rejected/              # 丢弃的数据
│       └── reports/               # 评估报告
└── README.md                      # 项目说明
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```
### 2. 配置模型

所有模型配置统一通过 CLI 参数指定，API 密钥未通过 `--llm-api-key` 指定时会自动从环境变量 `ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY` 读取。

#### CLI 参数一览

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--llm-provider` | 字符串 | LLM 提供商：`anthropic` / `openai` / `vllm` | `anthropic` |
| `--llm-model` | 字符串 | 模型名称 | `claude-3-5-sonnet-20241022` |
| `--llm-base-url` | 字符串 | 自定义 API 地址（vLLM/兼容接口使用） | — |
| `--llm-api-key` | 字符串 | API 密钥（未指定时从环境变量读取） | — |
| `--llm-temperature` | 浮点数 | 生成温度 | `0.1` |
| `--llm-max-tokens` | 整数 | 最大生成 token 数 | `1500` |
| `--total-threshold` | 整数 | 总分阈值 | `70` |
| `--accuracy-threshold` | 整数 | 准确性阈值 | `35` |
| `--checkpoint-interval` | 整数 | 检查点保存间隔 | `100` |
| `-o`, `--output` | 字符串 | 输出目录 | `data/output` |

#### 各提供商用法示例

**Anthropic（默认）**

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python -m src.main data/input/example_qa.json
```

**OpenAI**

```bash
export OPENAI_API_KEY="sk-..."
python -m src.main data/input/example_qa.json \
    --llm-provider openai \
    --llm-model gpt-4o
```

**vLLM 本地模型**

```bash
# 先启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model --port 8000

# 再运行评估
python -m src.main data/input/example_qa.json \
    --llm-provider vllm \
    --llm-model Qwen/Qwen2.5-7B-Instruct \
    --llm-base-url http://localhost:8000/v1
```

**自定义生成参数**

```bash
python -m src.main data/input/example_qa.json \
    --llm-temperature 0.2 \
    --llm-max-tokens 2000 \
    --total-threshold 75 \
    --accuracy-threshold 40
```

> 完整参数列表可通过 `python -m src.main --help` 查看。

### 3. 准备数据

将医学 QA 数据放入 `data/input/` 目录，支持 JSON、CSV、JSONL 格式。

示例格式：
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
python -m src.main data/input/example_qa.json -o data/output
```

> 数据加载阶段如果遇到空的 question/answer，会跳过该条并输出 warning 日志（不会导致整个批处理失败）。

### 5. 查看结果

输出文件结构：
```
data/output/
├── cleaned_data/
│   └── retained_qa.jsonl       # 保留的高质量数据
├── rejected/
│   └── discarded_qa.jsonl      # 丢弃的数据及原因
├── reports/
│   ├── evaluation_report.json  # 完整评估报告
│   └── summary.txt             # 可读摘要
└── summary.json                # 统计摘要
```

## 架构设计

详见 [docs/architecture-design.md](docs/architecture-design.md)

## 质量评估维度

| 维度 | 权重 | 说明 | LLM 评分依据 |
|------|------|------|-------------|
| 完整性 | 30% | 问题清晰、回答充分、信息完整 | 结合实体识别结果 |
| 准确性 | 45% | 医学正确、术语准确、符合指南 | **工具证据**：术语库匹配、指南符合性、百科验证 |
| 专业性 | 25% | 术语规范、表达专业、有免责声明 | 术语规范性、结构化表达 |

## 决策规则

代码基于 LLM 评分结果判断：

- **保留 (RETAIN)**：总分 ≥ 70 **且** 准确性 ≥ 35
- **丢弃 (DISCARD)**：总分 < 70 或 准确性 < 35

准确性是关键维度，必须达到阈值以上才保留。

> 阈值可通过 `--total-threshold` 和 `--accuracy-threshold` 调整。

## 开发计划

- [ ] 基础架构搭建
- [ ] 工具实现（EntityExtractor、TerminologyValidator、GuidelineChecker、WikipediaVerifier）
- [ ] 评分引擎实现
- [ ] 批量处理流程
- [ ] 报告生成

## 技术栈

- Python 3.10+
- Anthropic / OpenAI SDK (LLM 调用)
- scispaCy (医学NER)
- ICD-10 / SNOMED / UMLS (医学术语)
- vLLM / OpenAI 兼容接口 (本地模型部署)

---
