# DeepSeek V4 Flash Vision Exp · A100 · R3.3

面向 **8×A100 80GB** 的 DeepSeek-V4-Flash-Vision-Exp 部署源码。默认固定为已经现场使用的 **DSpark k6 greedy + V2 runner + Breakable CUDA Graph** 方案，所有协议共用唯一模型名 **`DeepSeek-V4-Flash`**。

本仓库和 Release 仅发布源码、补丁、配置、升级脚本与合成测试素材。**不包含 Docker 镜像、模型权重、wheel、编译产物或完整离线包。** 现有部署可以使用轻量升级入口；新机器需要另行准备运行镜像、模型权重、驱动、Docker 与 NVIDIA Container Toolkit。

| 固定项 | 配置 |
|---|---|
| 模型名 | `DeepSeek-V4-Flash`，区分大小写，只有一个名称 |
| 权重 | DeepSeek-V4-Flash-Vision-Exp，多模态 |
| 上下文 / 调度上限 | 262144 tokens / 32 请求 |
| GPU / 显存比例 | TP8 / 0.92 |
| 投机解码 | DSpark，6 个 draft tokens，greedy |
| 执行 | V2 runner、Breakable CUDA Graph，eager=false |
| 缓存 | 前缀缓存启用，FP8 KV，block=256 |
| 图片 | 计数上限 999，视觉 TORCH_SDPA |
| 协议 | Chat Completions、Responses、Anthropic Messages |
| 访问 | 宿主机及 Docker bridge，8005，后端免 API key |

## 现有 R3.1 / R3.2 升级

将本仓库的 `base64-r3.3.cmd` 复制到已有部署目录，在该目录执行：

```bash
bash base64-r3.3.cmd
```

脚本完整解码并校验后才执行；复用已校验的本机镜像和测试素材，准备新目录后再切换服务。升级需要重启；失败会按原运行状态回滚。可读控制器、安装器和逐文件校验清单位于 [upgrade/](upgrade/README.md)。它只接收已定型的 R3.1/R3.2 目录，不把任意旧 R3 目录当成可直接升级版本。

## 使用已准备好的 R3.3 镜像

只有在本机已有 `dsv4-flash-a100:20260914-r3.3` 且内容 ID 匹配时，源码目录可以直接作为部署目录：

```bash
cp deployment.env.example deployment.env
# 按实际位置修改 deployment.env 中的 MODEL_DIR。
bash preflight.sh
bash gpu_kernel_test.sh
bash start.sh
bash verify.sh
```

模型目录默认 `/ai/models/deepseek-v4-flash-vision-exp-modelscope`。`load_image.sh` / `deploy.sh` 是完整离线交付的入口，需要本仓库未附带的镜像 tar；仅下载 GitHub 源码不能执行镜像导入。源码锁定、补丁重建和本机构建说明见 [source/README.md](source/README.md)。

## 协议和缓存计费

宿主机基址 `http://127.0.0.1:8005`；Docker 客户端使用 `http://host.docker.internal:8005`，Linux 需要 `host.docker.internal:host-gateway` 映射。

| 接口 | 缓存读取字段 |
|---|---|
| `/v1/chat/completions` | `usage.prompt_tokens_details.cached_tokens` |
| `/v1/responses` | `usage.input_tokens_details.cached_tokens` |
| `/v1/messages` | `usage.cache_read_input_tokens` |

三个协议直接由同一 vLLM 服务处理，不需要额外协议转换代理。流式响应强制保留 usage；缓存字段来自真实统计，不伪造命中。New API 的渠道模型也使用 `DeepSeek-V4-Flash`；其用户鉴权、缓存倍率和账单展示仍由网关配置决定。详见 [NEWAPI.md](NEWAPI.md) 和 [客户端示例](clients/README.md)。这些接口测试不代表全部 Codex / Claude Code 功能均已认证。

## 验证与性能范围

`verify.sh` 提供 12 项多模态用例、5/8 图顺序、三协议 JSON/SSE 与工具、缓存 usage、10 轮短请求 C32、近 256K 单请求验收。`performance_test.sh` 保留单请求预热、重复长输出和投机指标采样。

前身 R3 方案四的现场样本中，1024-token 长输出的单请求 decode 中位数为文本约 **176 token/s**、带图约 **182 token/s**；使用反馈为与 R2.3 相近。这不是同机同步控制变量的 R2.3/R3.3 A/B 结论，R3.3 本身未重新执行目标 GPU 性能验收。

本次发布检查包括源码补丁回放、脚本语法、合成客户端判分、升级/回滚分支和容器配置；不将构建机静态检查标记为 GPU 推理通过。999 是图片计数配置上限，已有验收最多 8 图；短请求 C32 与近 256K 单请求分别测试，未验证 32 路同时满 256K。不开原生视频/音频。已报告的 Codex 经 New API 的 HTTP 500 根因未确认，R3.3 不宣称修复。详见 [已知限制](KNOWN_ISSUES.md)。

## 来源与许可证

基于 [haosdent/vllm](https://github.com/haosdent/vllm) 的固定 SM80 分支，移植 Vision 支持并保留 DSpark；固定 commit、补丁和 renderer 摘要见 [source/source-lock.json](source/source-lock.json)。模型来自 [DeepSeek / ModelScope](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp/files)。本项目脚本采用 Apache-2.0；第三方来源和授权见 [THIRD_PARTY.md](THIRD_PARTY.md)。
