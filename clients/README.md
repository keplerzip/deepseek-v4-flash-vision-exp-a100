# R3.3 客户端示例

所有配置仅使用 `DeepSeek-V4-Flash`，保留 262144 上下文。Codex 通过 Responses HTTP/SSE，Anthropic 客户端通过 Messages HTTP/SSE 连接同一服务。这里的示例不会自动安装或覆盖客户端配置。

模型后端免鉴权，不需要配置 API key。将示例中需要的配置合并到客户端现有文件。个别 SDK 如自行强制要求 key 参数，可用任意本地占位文本，后端不校验它。

客户端运行于服务宿主机时使用 `127.0.0.1:8005`；位于 Docker 时使用带 host-gateway 映射的 `host.docker.internal:8005`。若经过 New API，改为 New API 地址及其访问 token，模型名称仍为 `DeepSeek-V4-Flash`。

示例沿用此前 HTTP/SSE 配置方式。协议测试不是完整 Codex / Claude Code 会话兼容性认证；冻结服务的功能范围见根目录 README。
