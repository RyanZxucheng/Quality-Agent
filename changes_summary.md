# vLLM 支持功能变更摘要

## 变更文件列表

### 1. src/config.py
**变更**: 添加多LLM后端配置支持
- 新增配置字段: llm_provider, llm_model, llm_base_url, llm_api_key
- provider可选: "anthropic", "vllm", "openai"
- base_url用于vLLM本地部署，如"http://localhost:8000/v1"

### 2. src/scoring/llm_engine.py
**变更**: 重构为支持多后端LLM评分引擎
- 新增抽象基类LLMClient
- 实现AnthropicClient和OpenAICompatibleClient
- OpenAICompatibleClient支持vLLM、OpenAI及其他兼容API的服务
- LLMScoringEngine根据provider配置选择客户端
- 系统prompt和用户prompt模板优化

### 3. requirements.txt
**变更**: 添加依赖
- 新增openai>=1.0.0用于vLLM/OpenAI兼容接口

### 4. README.md
**变更**: 添加vLLM配置指南
- 添加方案B: vLLM本地部署配置说明
- 包含启动vLLM服务的命令示例
- 环境变量配置说明

### 5. src/main.py
**变更**: 命令行参数默认值
- accuracy_threshold默认值从25改为35（可能）

### 6. src/processor/batch_processor.py
**变更**: 导入LLMScoringEngine
- 从scoring.llm_engine导入LLMScoringEngine
- 处理流程: 证据收集 -> LLM评分 -> 决策

### 7. src/decision/engine.py
**变更**: 决策逻辑
- 使用准确性阈值判断
- 默认准确性阈值可能为35

## 核心功能变更
1. **多LLM后端支持**: 支持Anthropic Claude、vLLM本地部署、OpenAI兼容接口
2. **vLLM配置**: 通过base_url配置本地vLLM服务地址
3. **架构重构**: 从纯工具评分改为证据收集+LLM评分
4. **准确性权重提高**: 准确性维度权重45分，阈值35分

## 代码位置
项目根目录: E:\CC_fold\medical-qa-quality-agent\