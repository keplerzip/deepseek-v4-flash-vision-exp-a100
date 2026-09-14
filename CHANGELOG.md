# R3.3 · 2026-09-14

基于R3.2定稿，唯一模型名保持 `DeepSeek-V4-Flash`，Chat、Responses、Messages共用同一模型。只保留 DSpark k6 greedy / V2 / Breakable CUDA Graph 最终方案。

推理参数不变：0.92显存比例、262144总上下文、32并发、999图片计数上限、TP8、FP8 KV、前缀缓存及真实缓存usage。模型服务免API key，模型权重路径不变。未更新推理内核、依赖或tokenizer修复。

轻量升级识别R3.2和R3.1部署，支持已停止的旧容器；按固定镜像ID复用原始R3或R3.1/R3.2完整镜像。模型/缓存路径继续用显式参数传入安装器，兼容sudo清理环境变量。新候选准备完毕后才切换，失败按旧运行状态恢复。

用户要求停止进一步诊断并定稿。此前Codex经New API出现的“high demand”/HTTP 500根因未确认，本版本不宣称修复该问题；未更改New API运行配置或账单。目标机真实推理以部署后的verify.sh结果为准。
