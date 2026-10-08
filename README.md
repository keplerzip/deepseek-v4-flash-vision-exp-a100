# R3.9 — Codex 多图工具结果修复

固定 DeepSeek-V4-Flash-Vision-Exp 权重；唯一服务名称 `DeepSeek-V4-Flash`。
保留 1,048,576 上下文、并发 32、显存 0.92、图片上限 999、DSpark k6 / V2 / CUDA Graph、无后端 API key。

本版修复 Responses / Chat 工具结果中的图片被编码成 `[Unsupported image]`，导致 `Could not match all DeepSeek image placeholder tokens` 的问题。同时将工具结果排序提前到媒体收集前，使图像数据、UUID 与占位符保持一致。Anthropic 工具图片保留在原 tool_result 中，不再拆为独立用户消息。Responses 的工具输出图片也支持省略 `detail`，非法值仍会被拒绝。

版本对照表明该工具图片缺陷在 R3.7 与 R3.8 均存在；普通用户消息多图可以通过，因此不能仅由本次报错认定为 R3.8 新增回归。R3.9 保留原有内核与运行参数，不宣称 TPS 提升。

## GitHub 源码发行与升级

本仓库和 R3.9 Release 只发布可读源码、配置、来源记录及合成测试素材，不包含模型权重、Docker 镜像、wheel、编译缓存或编码升级载荷。新机器须另行准备兼容的冻结镜像和模型；本版 `install.sh` 用于已有 R3.8 部署的机器。

在现有项目目录中，与 R3.8 安装目录同级获取 R3.9 源码：

```bash
git clone --branch R3.9 --depth 1 https://github.com/keplerzip/deepseek-v4-flash-vision-exp-a100.git r3.9-1m-c32-20260929
cd r3.9-1m-c32-20260929
sha256sum --check --quiet FILES.sha256
sudo docker info >/dev/null && bash install.sh
```

离线机器可从 [R3.9 Release](https://github.com/keplerzip/deepseek-v4-flash-vision-exp-a100/releases/tag/R3.9) 下载 GitHub 自动生成的 Source code 归档，在旧 R3.8 目录旁解压；进入源码根目录后执行同样的文件校验和安装命令。GitHub 不上传部署压缩包或其他附件。发布不改变现有部署，已安装并通过验收的 R3.9 无需因源码发布而重启。

从 `dsv4-flash-r38` 的真实挂载发现旧目录，按实际镜像 ID 和标签验证并复用，不要求 R3.3 镜像标签。不下载依赖、权重或镜像。默认在旧目录同级创建 `r3.9-1m-c32-20260929`，不会嵌套在旧安装目录内。`install.sh` 先验证模型、CPU 回归和 GPU 环境，再停旧服务、启动并完整验收 R3.9。失败恢复原来运行或停止状态。成功标识是 `R39_INSTALL=PASS`。

新增图片工具历史验收包含 2/8 图、function/custom 工具输出、倒序返回、历史回放、JSON/SSE 和三个协议。它校验模型给出的图片顺序答案；没有实际运行时不会标记通过。

## 运维

在新的 R3.9 目录运行 `bash start.sh`、`bash stop.sh`、`bash verify.sh`。专项复测使用 `bash codex_image_test.sh`。`bash performance_test.sh` 可测单请求 TPS。`rollback.sh` 恢复安装时 R3.8 的运行状态。旧目录和编译缓存仍被保留，请勿删除缓存的真实目标目录。

构建机没有 A100，GPU 推理与真实 New API 转发须由目标部署验收。999 是允许的图片上限，不代表已实测同时输入999图；32 路满 1M 及 New API 账本计费均未验证。
