# DeepSeek V4 Flash Vision · A100 · R3.3

唯一模型名 `DeepSeek-V4-Flash`，区分大小写。模型服务免 API key；宿主机及 Docker 内网可直接访问。所有协议共用一个模型服务。

固定 DSpark k6 / greedy、V2 runner、Breakable CUDA Graph（enforce_eager=false）、TP8、显存比例0.92、上下文262144、并发32、图片计数上限999、FP8 KV、block256、batch4096、bf16、视觉TORCH_SDPA和前缀缓存。不开原生视频/音频。

模型权重放在 `/ai/models/deepseek-v4-flash-vision-exp-modelscope`；仅在 `deployment.env` 修改实际模型路径。权重不随包复制。需已安装兼容 Docker、NVIDIA Container Toolkit 和驱动；默认检查8张A100 80GB及驱动>=580.126.20。

完整镜像版含所有镜像父层，不需要拉取基础镜像：解压后执行 `bash deploy.sh`。现有服务升级请使用附带的轻量 Base64 升级文件；它识别R3.2/R3.1部署，复用已校验的现有镜像和测试素材，不传输大镜像，不重下载模型。

以后在本目录执行 `bash start.sh`、`bash stop.sh`、`bash verify.sh`，不带方案编号。完整 verify 包括12项图片用例、5/8图顺序、三协议JSON/SSE与工具、真实缓存 usage、10轮32并发、近256K单请求。999是配置计数上限，不等于999图实测；单请求长上下文与短请求32并发不等于32路同时满256K。

宿主机 root URL `http://127.0.0.1:8005`，Docker 内使用 `http://host.docker.internal:8005`，Linux容器需 host-gateway 映射。Chat `/v1/chat/completions`，Responses `/v1/responses`，Messages `/v1/messages`，另有 `/v1/messages/count_tokens`、`/v1/completions`、`/v1/models`。New API 渠道模型名同步为 `DeepSeek-V4-Flash`，缓存计费字段与网关复测见 `NEWAPI.md`。

从R3.2或R3.1升级时保留旧目录和容器，成功后旧容器停止；失败停止新候选并恢复先前运行状态。升级会重启模型服务，内存前缀缓存重新建立；磁盘编译缓存复用。已有New API本身的客户端鉴权/计费配置不由本包修改。

本机发布文件校验不代表目标机R3.3 GPU验收已通过；以运行后的验收报告为准。客户端示例不是全部Codex/Claude Code功能认证。

此次为R3.2部署方案定稿；已报告的Codex经New API的HTTP 500尚未定位，不宣称已修复，见 `KNOWN_ISSUES.md`。
