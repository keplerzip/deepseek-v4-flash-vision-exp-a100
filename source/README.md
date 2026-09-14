# 源码与构建

`source-lock.json` 固定 R2.3 的 SM80 基线、Vision 移植补丁及最终线程安全 renderer。`r3-port-from-r2.3.patch` 涵盖 39 个文件；已回放到冻结 R2.3 源码，结果与 R3 源码快照对应文件一致。运行时还需要覆盖 `deepseek_v4_renderer.py`，并使用 `native/` 中的源码编译视觉路由扩展。

本目录只包含补丁、源文件、依赖版本清单、许可证和模型元数据（不含权重）。`python-dependencies.json` 记录已有运行环境，不是声称可在任意系统直接安装成功的依赖锁文件。

联网开发机可以还原固定源码：

```bash
bash source/prepare_source.sh
```

它在独立 `work/` 目录拉取固定 commit、校验并应用补丁、覆盖 renderer，然后生成本地 `source/vllm-source.tar.gz`。生成物被 Git 忽略，不随源码 Release 上传。当前仅校验过补丁对冻结源文件的回放；重新联网拉取和完整 CUDA 编译需在构建环境执行。

已有已校验 R3.3 运行镜像及 Docker 权限时，可以复用其中的编译器与依赖：

```bash
cp deployment.env.example deployment.env
bash source/prepare_source.sh
bash source/rebuild_image.sh
```

构建产物是独立标签 `dsv4-flash-a100:r3.3-local-rebuild`。它不会替换冻结镜像，也不会自动加入部署镜像 ID 允许列表；需要独立 GPU 验证。完整宿主安装、权重下载以及从零构建 CUDA/PyTorch 依赖不包含在这一入口中。

`Dockerfile` 记录 R3.2 → R3.3 的版本元数据层，R3.3 没有新增内核或依赖修改。历史来源保存在 `provenance.json`，其中旧版本模型拼写仅作来源记录，不是当前服务别名。
