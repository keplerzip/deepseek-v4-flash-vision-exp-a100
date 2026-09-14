# 单模型与 New API

唯一上游模型名为 `DeepSeek-V4-Flash`，大小写固定。实际权重仍为 DeepSeek V4 Flash Vision Exp。Chat、Responses、Messages 共用同一个后端模型和前缀缓存池，不注册 -claude、-vision-exp 等别名。不同协议模板或图片内容可能形成不同前缀，不能保证跨协议命中。

沿用宿主 127.0.0.1:8005 和 Docker bridge 的 8005 入口。Docker New API 使用 `host.docker.internal:8005`，Linux 容器需要 `host.docker.internal:host-gateway` 映射。模型服务免 API key，直连请求不需要 Authorization 或 x-api-key；New API 自身用户 token 策略独立。

| 原生路径 | 内容 | 应保留的 usage |
|---|---|---|
| /v1/chat/completions | Chat、图片、工具、JSON/SSE | prompt_tokens_details.cached_tokens |
| /v1/responses | Responses、图片、function_call、JSON/SSE | input_tokens_details.cached_tokens |
| /v1/messages | Anthropic Messages、图片、tool_use、JSON/SSE | cache_read_input_tokens / cache_creation_input_tokens |
| /v1/messages/count_tokens | Anthropic 输入计数 | input_tokens |
| /v1/completions | 文本补全 | 原生 completion usage |

显式开启 `--enable-prefix-caching`、`--enable-prompt-tokens-details`、`--enable-force-include-usage`。Chat 即使请求未带 stream_options.include_usage 也返回最终 usage；测试同时覆盖显式 include_usage 和省略参数。Responses 使用 response.completed 的 usage；Messages 合并 message_start 与 message_delta 的 usage。不要把每个流式片段的累计值反复相加。

Chat/Responses 的 input/prompt token 总数已经包含缓存读取；Anthropic 的 input_tokens 不包含 cache_read/cache_creation，合计时才相加。不会将所有未命中 token 伪报成缓存写入，也不会为了 UI 出现命中而造数。


New API 需启用相应 Responses/Messages 路由，避免多拼接 /v1。模型渠道只配置 `DeepSeek-V4-Flash`；上游基址是否加 /v1 按渠道拼接规则决定。标准 cached_tokens 字段可供 OpenAI 渠道读取，不依赖旧名称触发特殊 DeepSeek 字段转换。前端缓存折扣、倍率和消费日志仍由 New API 配置决定。

升级默认只自动验收后端。需要在当前 New API 实测时，将模型 token 写入一个权限600的文件，再执行：

```bash
R33_NEWAPI_URL=http://127.0.0.1:3000 R33_NEWAPI_TOKEN_FILE=/absolute/path/newapi-token bash newapi_test.sh
```

测试会发出真实请求并产生该网关的正常用量，校验三协议冷热缓存 JSON/SSE、多图和工具往返。结果记录返回 request id、真实 usage；与 New API 消费日志的缓存读取量核对才能判断账单链路。测试不读取或更改网关数据库/管理员配置。

图片计数上限 999，包括当前请求提交的历史图片。5/8图顺序验收不等于999图容量认证；实际还受262144总上下文、图片大小和视觉编码资源限制。
