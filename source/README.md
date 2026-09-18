# R3.4 运行源码

`overlay/` 是对冻结基础镜像的只读文件覆盖。`r34-runtime.patch` 记录相对 R3.3 实际运行文件的改动，包含 SPDX/Apache-2.0 声明。`overlay-manifest.json` 为每个覆盖文件的 SHA-256。

`prepare_source.sh` 从本机的已校验镜像导出 Python 文件，再应用本版本 overlay；不会请求上游不存在的派生提交。它不导出 CUDA 二进制或权重，不声称可从公共上游提交独立全量重建旧 CUDA 内核。

可选 `rebuild_image.sh` 仅叠加覆盖源码，基础层复用，不下载任何内容。默认部署使用原镜像加 overlay。

保留 `r3-port-from-r2.3.patch`、`native/`、原始依赖清单和模型元数据供历史审计。模型元数据中的原始最大上下文仅记录检查点原貌，服务固定 262144，不启用其原始最大窗口。
