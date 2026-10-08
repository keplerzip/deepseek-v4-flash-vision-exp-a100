# R3.9 冻结镜像覆盖源码

`overlay/` 的 25 项 Python 文件通过只读挂载覆盖目标机 R3.8 容器的实际冻结镜像。`overlay-files.txt` 指定挂载表，`overlay-manifest.json` 给出逐项 SHA256；`r39-runtime.patch` 记录本版相对 R3.8 的四项运行时修复，`r38-runtime.patch` 与更早差异保留为继承来源记录。基础镜像内的 Python/CUDA 依赖和模型权重未改动。

`Dockerfile` 与 `rebuild_image.sh` 仅供开发机可选地构建派生镜像；**增量部署不运行它们，也不携带派生镜像**。`r37-runtime.patch` 和更早差异保留作历史审计。模型元数据的 1,048,576 最大位置长度与 `config/service.json` 的服务窗口一致；现场推理仍须通过安装验收。
