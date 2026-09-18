# 轻量离线升级

`controller.sh.in` 校验旧容器、冻结镜像和挂载，安装到独立候选目录，通过 CPU 检查后切换服务。GPU 验收失败时回滚原有运行/停止状态。

`installer.py` 校验完整 payload 的每个 SHA-256 后写入，模型路径、缓存路径、镜像 ID 均为显式参数。不复制权重，不要求 API key。测试素材全部内嵌，不依赖旧 R3 的素材版本。

`build_command.py` 从同一源码树生成 `payload.json`、`upgrade-r34.sh` 和 `base64-r3.4.cmd`。Base64 文件用分行 heredoc，避免 Linux 单个命令参数 128 KiB 上限，完整解码校验后才执行。编码不是加密。

公开仓库只保留这里的可读生成器、安装器和控制器源码。生成后的编码载荷、命令文件及 `delivery.json` 仅用于本地交付，已加入 `.gitignore`，不包含在 R3.4 源码 Release 中。

```bash
python3 upgrade/build_command.py
```

交付机器只需在原部署目录执行 `bash base64-r3.4.cmd`。也可用 `R34_SOURCE_DIR` 显式指定。源码仓库用于审计与重建，不是原部署目录。
