unset SSL_CERT_FILE
unset REQUESTS_CA_BUNDLE

python -m src.main data/input/example_qa.json \
    --llm-provider vllm \
    --llm-model qwen3dot5_4b \
    --llm-base-url http://14.103.165.117:8010/v1