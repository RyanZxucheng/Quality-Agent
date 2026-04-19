#!/bin/bash

python -m src.main \
  --dataset "data/input/example_qa.json" \
  --llm-provider "vllm" \
  --llm-model "Qwen3.6-35B-A3B" \
  --llm-base-url "http://localhost:9060/v1" \
  --max-retained 100 \
  --batch-size 5