# DeepSeek-V4-Flash-Vision-Exp on A100 — R3.4

R3.4 复用已经导入的 R3 / R3.1 / R3.2 / R3.3 镜像及 Vision-Exp 权重，通过只读源码覆盖部署。无需重新传输大镜像、下载依赖或修改模型文件。唯一服务名为 **`DeepSeek-V4-Flash`**，全部协议共用同一个引擎。

2026-09-16 的部署端反馈已通过升级验收，2026-09-18 确认方案有效并发布源码。单请求文本 / 图像端到端中位速度约为 **210 / 203 token/s**，与该机器升级前基本持平；不宣称显著提速。详细数据与验证范围见 [VALIDATION.md](VALIDATION.md)。

| 配置 | 固定值 |
| --- | --- |
| 硬件 | 8 × A100 80GB，TP8 |
| 推理 | DSpark k6 greedy，V2 runner，Breakable CUDA Graph |
| 上下文 / 并发 | 262144 / 32 |
| 显存比例 | 0.92 |
| 多模态 | 开启，图片数量上限 999，TORCH_SDPA |
| 缓存 | FP8 KV，prefix cache，真实 usage/cache 字段 |
| 接口 | Chat Completions、Responses、Anthropic Messages / count_tokens |
| 服务鉴权 | 不设 API key |
| 地址 | 本机 127.0.0.1:8005、Docker bridge gateway:8005 |

## 从部署机器上的 R3 升级

将交付的 `base64-r3.4.cmd` 放入原部署目录，执行：

```bash
bash base64-r3.4.cmd
```

文件由 Bash 执行；`.cmd` 沿用交付命名，并非 Windows 命令。文件使用 Base64 编码和 gzip 压缩，在执行任何升级操作前完整解码并校验 SHA-256。也可解压小型升级包，直接执行其中的 `upgrade-r34.sh`，工作目录仍为原部署目录。

入口支持旧服务运行、停止或容器不存在三种状态。优先识别运行的已知容器，复用其模型/编译缓存挂载，并校验冻结镜像内容 ID。只有旧目录且无容器时，从 `deployment.env` 读取路径；未设置模型路径时使用 `/ai/models/deepseek-v4-flash-vision-exp-modelscope`。如有多个旧部署，可用 `R34_SOURCE_DIR=/绝对路径 bash base64-r3.4.cmd` 指定。

升级先创建独立候选目录，校验文件、模型索引/权重头与 CPU 回归，再停止旧服务。随后执行 GPU 数值检查、启动、完整验收与测速。旧服务运行时会先用同一脚本测文本/图像 C1 基线；完整的三次 1024-token 样本若显示新版本中位吞吐下降超过 20%，阻止切换定稿。缺失基线或提前 EOS 时明确标记不可比，不声称提速。失败时停止候选版本，并恢复原容器的运行/停止状态；保留日志。**不会自动降低显存比例、关闭多模态或将 k6 改为其他值。**

成功输出 `R34_UPGRADE=PASS INSTALL_DIR=... REPORT=...`。以后在 `INSTALL_DIR` 中运行 `start.sh` / `stop.sh` / `verify.sh`，不带方案编号。原 R3 容器、目录和镜像保留供回滚，升级目录内的 `rollback.sh` 可恢复此前运行/停止状态；确认新版本后再决定磁盘清理。升级过程中服务会中断，内存前缀缓存会重新预热，磁盘编译缓存继续复用。

## 实际变化

- DSpark 辅助隐藏状态复用前一层融合 mHC 结果，保留本分支 int8 all-reduce 的 `x_scales` 和五项返回协议；避免无用 MTP buffer 拷贝。
- 将三层草稿的上下文 WKV 投影合并为一次量化线性计算。仍加载原检查点的权重和 scales，保留每层归一化、RoPE、KV 写入。每个 TP rank 在真实量化打包后，用非零探针与原计算路径比对，失败即终止启动，不静默退回。
- Responses 不支持的输入类型使用现有 `VLLMValidationError` 返回 400；支持回放空 assistant 输出和多个文本输出块。没有增加 `computer_call` 或加密 reasoning 的支持。
- 原 R3 一步升级，不再要求事先存在运行中的方案四，也不依赖环境变量向 Docker 隐式传递模型路径。
- 修正源码准备入口：从验证过的已有镜像导出 Python 源码并应用覆盖，不再 fetch 远端不存在的本地派生提交。

移植依据与未采纳项目见 [变更说明](CHANGELOG.md)。k3 对照未纳入本版本。

## 验证与边界

本地 CPU/交付流程及部署端反馈见 [VALIDATION.md](VALIDATION.md)。构建机没有目标 A100 和完整模型；实际推理结果来自部署端提供的日志。部署端已反馈 `R34_UPGRADE=PASS`，该成功标志要求以下验收关卡通过：

- mHC 数值与 CUDA Graph 重放检查、已有 Vision GPU 路由检查；加载后 8 个 TP rank 的量化 WKV 数值对比。
- 完整多模态用例、工具往返、流式三协议、Responses 错误码和空历史回复测试。
- 三协议的真实缓存字段及换图负对照，10 轮 32 并发，单请求近 256K 图文上下文。
- 文本和图像 1024-token 单请求测速，保留原始使用量与耗时。

本次比较基线为该部署机器原先运行的 R3.1，不能当作 R3.3 → R3.4 的受控对照。三次完整样本的文本端到端中位速度变化为 -1.74%，图像为 +2.24%；样本量不足以证明稳定提速。性能比较中的 `PASS` 表示未下降超过 20%，不表示一定加速。

999 是请求图片数量上限，仍受分辨率、token 预算和显存约束，不等于验收过 999 张任意图片。C32 与近 256K 为分别测试，不等于 32 路满窗口。New API 原始转发可用 `newapi_test.sh` 验证，实际账单仍需核对你所安装的 New API 版本；不伪造缓存 token 或修改 New API 数据库。详见 [NEWAPI.md](NEWAPI.md)。

## 源码与镜像

`config/service.json` 是唯一部署配置。模型路径写在 `deployment.env`，不包含密钥。`source/overlay/` 为完整可审计的覆盖源码，`source/r34-runtime.patch` 为相对 R3.3 实际运行文件的差异。

```bash
python3 upgrade/build_command.py
```

此命令可从源码重新生成小型升级入口；内容和素材全部嵌入，无需旧测试素材。`source/rebuild_image.sh` 可选生成复用基础层的本地派生镜像用于归档。默认部署仍通过原镜像 ID 和源码覆盖运行。基础镜像内的 CUDA、Torch、vLLM 原生二进制没有重新编译；不是从纯上游源码全量构建依赖。

GitHub 仓库与 Release 附件仅包含可读源码、配置和合成测试素材。Base64 命令、编码后的 `upgrade/payload.json` 和交付哈希元数据仅供本地生成，不纳入本次 GitHub 发布；模型权重、运行镜像和全依赖离线包也不随源码发行。从源码本地生成的入口位于 `upgrade/base64-r3.4.cmd`；将它复制到已有部署机器的原部署目录后执行。已有 R3.4 且验收通过的部署无需为本次源码发布重新升级。
