# EdgeMemory · 端侧多模态视频记忆

[English](README.md)

EdgeMemory 面向 Qualcomm Dragonwing IQ-9075，在设备本地将视频转化为可检索、可追溯的语义事件，支持自然语言搜视频、以图搜视频、证据复审、带时间戳的问答与抽样摘要。

**技术特点：** VPU 解码、QIM/GLES GPU 预处理、GenieX HTP 双 4B 生成式模型、CLIP 512D + MiniLM 384D 双通道记忆、SQLite 和 Web 工作台。

真机修复后，734.167 秒视频在 967.759 秒内完成 196 条事件索引，本轮索引零失败；单独的一次 HTP LLM 证据问答耗时 3.498 秒，TTFT 723 ms，Decode 15.96 tokens/s。不同实验不代表同一次并发负载下的综合性能。

- [快速开始](docs/QUICKSTART.md)
- [架构说明](docs/ARCHITECTURE.md)
- [性能口径](benchmarks/README.md)
- [贡献指南](CONTRIBUTING.md)

这是工程预览版本，不是已经验证多路实时运行的商用 NVR。源码采用 MIT 许可；外部模型和 SDK 不随源码分发。
