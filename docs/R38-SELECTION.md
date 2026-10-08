# R3.8 上游选取 — 2026-09-23

针对冻结 Vision-Exp / A100 / DSpark k6 路径，仅吸收可用现有源码覆盖验证的协议正确性修复：

| vLLM PR | 状态（核对时） | R3.8 处理 |
| --- | --- | --- |
| [#46257](https://github.com/vllm-project/vllm/pull/46257) | open | 对话模板支持 `add_generation_prompt` 与 `continue_final_message`，保留旧版 inline system 行为 |
| [#53739](https://github.com/vllm-project/vllm/pull/53739) | open | 截断流式工具参数时补齐已输出 JSON，避免无法解析的残片 |
| [#58291](https://github.com/vllm-project/vllm/pull/58291) | open | request tools 除 system 外也可附着已有 developer 消息 |
| [#58296](https://github.com/vllm-project/vllm/pull/58296) | open | 多段纯文本工具结果以空行连接，与 DeepSeek 编码器一致 |
| [#58295](https://github.com/vllm-project/vllm/pull/58295) | open | 工具调用前移除恰好一个模板分隔符，避免客户端回传历史时重复空行 |

未采用 [#57632](https://github.com/vllm-project/vllm/pull/57632) DSpark 全步 CUDA Graph：公开测试针对 V4.1/GB200/k5，C32 收益有限；目标 1M 窗口可能增加捕获显存。未采用 [#56694](https://github.com/vllm-project/vllm/pull/56694) W2 top-k 剪枝：变动跨 DSpark 权重和内核，冻结 Vision-Exp/A100 上无等价正确性与性能证据。V4.1 专属 vision profiler 与 ROCm 路径也不直接适用。New API 网关的缓存展示改动仍需独立网关版本和实际账本验证，本包不擅自改写网关。

这些 PR 在核对时尚未合并，R3.8 只是带回归测试的定向移植，并不宣称上游正式版本认证。原有 DSpark k6/V2/Breakable Graph 保持默认；性能变化需目标机实测。
