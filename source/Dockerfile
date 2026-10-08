ARG BASE_IMAGE=dsv4-flash-a100:20260914-r3.3
FROM ${BASE_IMAGE}
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
LABEL com.deepseek.release="2026.09.29-r3.9" \
      com.deepseek.model="DeepSeek-V4-Flash" \
      com.deepseek.runtime="DSpark-k6-V2-BreakableGraph"
