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

项目支持三种 LLM 提供商，可通过**环境变量**或**代码配置**两种方式设置。

#### 2.1 配置选项说明

| 配置项 | 类型 | 说明 | 默认值 |
|--------|------|------|--------|
| `llm_provider` | `LLMProvider` 枚举或字符串 | LLM 提供商，可选：`"anthropic"`, `"vllm"`, `"openai"` | `"anthropic"` |
| `llm_model` | 字符串 | 模型名称，取决于提供商 | `"claude-3-5-sonnet-20241022"` |
| `llm_base_url` | 字符串 | 自定义 API 地址（仅 vLLM 或自定义 OpenAI 兼容接口需要） | 空字符串 |
| `llm_api_key` | 字符串 | API 密钥（如未设置，会根据 provider 自动使用对应环境变量） | 空字符串 |

#### 2.2 各提供商详细配置

##### **Anthropic (默认)**
- **API Key 环境变量**: `ANTHROPIC_API_KEY`
- **支持模型示例**:
  - `"claude-3-5-sonnet-20241022"` (默认)
  - `"claude-3-opus-20240229"`
  - `"claude-3-haiku-20240307"`
- **依赖**: `anthropic` (已包含在 requirements.txt)

##### **OpenAI**
- **API Key 环境变量**: `OPENAI_API_KEY`
- **支持模型示例**:
  - `"gpt-4o"`
  - `"gpt-4-turbo"`
  - `"gpt-3.5-turbo"`
- **依赖**: `openai` (已包含在 requirements.txt)

##### **vLLM (本地部署)**
- **API Key 环境变量**: `OPENAI_API_KEY` (通常可留空)
- **base_url**: `"http://localhost:8000/v1"` (默认 vLLM OpenAI 兼容接口地址)
- **模型名称**: 你本地部署的模型名称，如 `"Qwen/Qwen2.5-7B-Instruct"`, `"meta-llama/Llama-2-7b-chat-hf"`
- **启动 vLLM 服务**:
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model /path/to/your/model \
      --port 8000 \
      --tensor-parallel-size 1
  ```

#### 2.3 环境变量配置（最简单）

根据选择的提供商，设置对应的环境变量：

```bash
# 使用 Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# 或使用 OpenAI
export OPENAI_API_KEY="sk-..."

# vLLM 通常不需要 API Key，但也可设置
export OPENAI_API_KEY=""
```

设置后，项目会自动使用默认配置：
- **提供商**: `anthropic` (如果设置了 `ANTHROPIC_API_KEY`)
- **模型**: `claude-3-5-sonnet-20241022`

#### 2.4 代码配置（更灵活）

如需自定义提供商、模型或配置，可在代码中通过 `AppConfig` 设置：

```python
from src.config import AppConfig, set_config

# 创建配置
cfg = AppConfig(
    llm_provider="openai",          # "anthropic", "vllm", "openai"
    llm_model="gpt-4o",             # 模型名称
    llm_base_url="",                # vLLM 服务地址，如 "http://localhost:8000/v1"
    llm_api_key="",                 # 如留空，会自动从环境变量读取
)

# 应用配置
set_config(cfg)
```

#### 2.5 完整配置示例

##### **示例 1: 使用 Anthropic Claude**
```python
from src.config import AppConfig, set_config

cfg = AppConfig(
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022",
    # llm_api_key 会自动从 ANTHROPIC_API_KEY 环境变量读取
)
set_config(cfg)
```

##### **示例 2: 使用 OpenAI**
```python
from src.config import AppConfig, set_config

cfg = AppConfig(
    llm_provider="openai",
    llm_model="gpt-4o",
    # llm_api_key 会自动从 OPENAI_API_KEY 环境变量读取
)
set_config(cfg)
```

##### **示例 3: 使用 vLLM 本地模型**
```python
from src.config import AppConfig, set_config

cfg = AppConfig(
    llm_provider="vllm",
    llm_base_url="http://localhost:8000/v1",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    # vLLM 通常不需要 API Key
    llm_api_key="",
)
set_config(cfg)
```

##### **示例 4: 使用自定义 OpenAI 兼容接口**
```python
from src.config import AppConfig, set_config

cfg = AppConfig(
    llm_provider="openai",           # 或 "vllm"
    llm_base_url="https://api.your-service.com/v1",
    llm_model="your-model-name",
    llm_api_key="your-api-key",
)
set_config(cfg)
```

#### 2.6 配置优先级

1. **代码中传递的参数**（如 `LLMScoringEngine(provider="openai", ...)`）优先级最高
2. **`AppConfig` 配置**（通过 `set_config()` 设置）次之
3. **环境变量**（`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`）为默认后备

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

> 说明：数据加载阶段如果遇到空的 question/answer，会跳过该条并输出 warning 日志（不会导致整个批处理失败）。

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

> **注意**:
> - 命令行参数可以覆盖阈值（见 `src/main.py`）：
>   - `--total-threshold`（默认 70）
>   - `--accuracy-threshold`（默认 35）
> - 如需要不同的阈值，可在运行命令中指定，如 `--accuracy-threshold 40`

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
