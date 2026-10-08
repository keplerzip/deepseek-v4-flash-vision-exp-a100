# R3.7 选取记录 — 2026-09-22

评估基线是 R3.6，实际权重始终为 DeepSeek-V4-Flash-Vision-Exp。复用冻结镜像与依赖，保持 DSpark k6、TP8、256K、C32、0.92、999 图片上限及唯一模型名 DeepSeek-V4-Flash。

| 上游 | 观察状态 | 纳入方式 |
| --- | --- | --- |
| [vLLM #57814](https://github.com/vllm-project/vllm/pull/57814) | Open | 修复 TokenizeChatRequest 中不可 pickle 的 tool_calls 迭代器，兼容 reasoning_content。增加调用方输入不变、非法工具参数与序列化检查。 |
| [vLLM #57885](https://github.com/vllm-project/vllm/pull/57885) | 9 月 21 日合并 | SWA 标记原地写入，压缩序列长度只在 prefill 时计算。没有引入新硬件指令或主机同步。 |
| [vLLM #57770](https://github.com/vllm-project/vllm/pull/57770) | Open | 将两阶段 Triton argmax 适配到冻结版 DSpark greedy 六步选词。保留首索引、NaN、int64 语义，不改变草稿数或验证算法；不支持的布局与 dtype 使用 Torch。 |

具体 head / merge SHA 在 source/provenance.json 固定，不跟随远程分支自动变化。此前 19 个 overlay 文件保持逐字节不变，新增 5 个覆盖文件。新增 GPU 路径须在目标 SM80 上通过 23 项数值与变输入 Graph 检查后才启动候选。

上游 argmax 的服务数据来自 GB200、V4.1、k5 和合成接受率；隔离算子的倍数不能换算成本部署加速。稀疏元数据改动的上游整体性能基本持平。R3.7 必须以现场前后测速判断，当前没有 A100 提速结论。

## 暂缓项及原因

- [#48416](https://github.com/vllm-project/vllm/pull/48416) / [#57592](https://github.com/vllm-project/vllm/pull/57592)：确实可复现 nullable / 缺省类型检查缺口，但配套调查发现冻结 StructuredOutputManager 沿用首次选择的后端。单独加强前端 auto 回退仍会通过错误后端编译。上游 [#43920](https://github.com/vllm-project/vllm/issues/43920) 与 [#49625](https://github.com/vllm-project/vllm/pull/49625) 也记录此限制，后者采用拒绝混合后端而非动态切换。暂不扩展引擎结构，也不把前端单测通过当作整个链路已修复。保留 R3.6 后端源码和现有能力边界。
- [#57856](https://github.com/vllm-project/vllm/pull/57856)：custom all-gather 涉及 TP8 通信和 IPC/Graph 适配，当前缺少 SM80 证据。
- [#58016](https://github.com/vllm-project/vllm/pull/58016)：冷启动权重读取过滤需要验证权重别名与加载完整性，不属于稳态 decode 加速。
- [#56625](https://github.com/vllm-project/vllm/pull/56625)：Vision encoder Graph 的新入口在 V4.1 wrapper，当前 Vision-Exp / TORCH_SDPA 仍需额外适配。
- SM100 / GFX950 融合实现不直接搬到 A100；不进行 k3 对照，不更新模型或二进制依赖。
- New API rc.40 的多模态转换修复属于独立网关组件，说明见 NEWAPI.md。未验证或更改实际网关和账本。

保留当前 Responses custom 文本工具兼容行为；未纳入上游后来对 grammar 格式的强制拒绝。当前 custom grammar 只是描述提示，不是受约束解码保证。
