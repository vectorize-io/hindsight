---
title: "在 Linux ARM64 上运行 Hindsight CLI 而无需解决方法"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [cli, linux, arm64, guide]
description: "使用新的发布资产在 Linux ARM64 上运行 Hindsight CLI，然后在 Pi 或 ARM 主机上配置配置文件、API 访问和日常内存命令。"
image: /img/guides/guide-run-hindsight-cli-on-linux-arm64.png
hide_table_of_contents: true
---

![在 Linux ARM64 上运行 Hindsight CLI 而无需解决方法](/img/guides/guide-run-hindsight-cli-on-linux-arm64.png)

如果您想**在 Linux ARM64 上运行 Hindsight CLI**，设置最终变得简单了。最新的发布流程现在提供了一流的 `hindsight-linux-arm64` 资产，这意味着 Raspberry Pi 盒子、Graviton 实例和小型 ARM 服务器不再需要本地重建或非官方复制步骤只是为了让 CLI 运行。如果您在工作时想要周围的文档，请保持打开 [CLI 参考](https://hindsight.vectorize.io/sdks/cli)、[安装指南](https://hindsight.vectorize.io/sdks/developer/installation)、[快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart) 和 [文档首页](https://hindsight.vectorize.io)。

<!-- truncate -->

## 快速答案

- Linux ARM64 现在包含在已发布的发布资产中，与现有的 AMD64 和 macOS 二进制文件一起。
- 最快的路径是下载 `hindsight-linux-arm64`、标记为可执行并将其移到您的 PATH 上。
- 安装二进制后，常规 `configure`、`bank`、`retain` 和 `recall` 命令的工作方式与其他平台上相同。

## 为什么此更新很重要

Linux ARM64 支持很重要，因为许多自托管 Hindsight 部署恰好登陆到该硬件类。壁橱里的 Raspberry Pi、廉价的 ARM VPS 或 AWS Graviton 实例通常足以用于轻量级内存服务，特别是如果您遵循较新的 [安装指导](https://hindsight.vectorize.io/sdks/developer/installation) 并根据精简映像或外部提供者来调整大小。

在此发布资产被连接到发布作业之前，即使其余平台运行良好，CLI 本身也很容易被忽视。新资产弥补了这一差距。这是一个小的发布工作流更改，但它使 ARM64 成为真正支持的路径而不是接近。

## 安装 Linux ARM64 二进制

直接使用已发布的发布资产：

```bash
curl -L   -o hindsight   https://github.com/vectorize-io/hindsight/releases/latest/download/hindsight-linux-arm64

chmod +x hindsight
sudo mv hindsight /usr/local/bin/hindsight
hindsight --help
```

如果您更喜欢在主目录中保留本地工具，请将二进制移到 `~/.local/bin` 中，并确保该目录在您的 PATH 上。关键检查很简单：`hindsight --help` 应该打印命令树而不是架构或权限错误。

## 为云或本地 API 访问配置 CLI

安装后，将 CLI 指向您实际想要使用的 API。

```bash
# Managed cloud
hindsight configure   --api-url https://api.hindsight.vectorize.io   --api-key YOUR_API_KEY

# Or a local deployment
hindsight configure --api-url http://localhost:8888
```

如果在环境之间切换，请使用 [CLI 参考](https://hindsight.vectorize.io/sdks/cli) 中的命名配置文件，而不是一遍又一遍地重写一个共享配置文件：

```bash
hindsight profile create prod   --api-url https://api.hindsight.vectorize.io   --api-key YOUR_API_KEY

hindsight -p prod bank list
```

环境变量仍然优先于配置文件，因此 CI 作业可以导出 `HINDSIGHT_API_URL` 和 `HINDSIGHT_API_KEY` 而不会与本地默认值冲突。

## 端对端验证核心工作流

配置 CLI 后，使用小型库和简单的内存往返来测试整个路径：

```bash
hindsight bank list
hindsight memory retain test-bank "Alice prefers async updates"
hindsight memory recall test-bank "How should I update Alice?"
```

如果这有效，ARM64 故事就完成了。您正在使用与 [保留 API 指南](https://hindsight.vectorize.io/sdks/api/retain)、[回忆 API 指南](https://hindsight.vectorize.io/sdks/api/recall) 和 [快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart) 中记录的相同内存命令，只是从 Linux ARM64 主机。

## 排除常见 Linux ARM64 错过

有一些失败值得首先检查：

- **`Exec format error`** 通常意味着您下载了错误的资产。双重检查文件名是 `hindsight-linux-arm64`，而不是 AMD64 构建。
- **`Permission denied`** 意味着二进制文件缺少执行位。重新运行 `chmod +x hindsight`。
- **Connection refused** 通常意味着您的本地 API 还没有启动，或者您将 CLI 指向了错误的主机和端口。
- **401 或 403 响应**通常意味着 API 密钥丢失、无效或针对错误的 Hindsight 环境。

如果 CLI 本身有效但回忆速度缓慢或主机感觉 RAM 紧张，请根据安装文档中的新 [硬件指导](https://hindsight.vectorize.io/sdks/developer/installation) 比较您的盒子。这通常是一个部署大小问题，而不是 CLI 问题。

## 常见问题

### 这是否替换安装脚本？

不。发布资产只是使 Linux ARM64 成为一个干净的下载目标。如果您的安装流程已经包装已发布的发布资产，此更改就是使 ARM64 干净地融入该路径的原因。

### 我可以针对 Hindsight Cloud 和本地服务器使用 CLI 吗？

是的。使用配置文件或环境变量。这是在云、暂存和本地部署之间切换的最干净的方式。

### ARM64 仅用于开发吗？

不。对于小型和中型工作负载，这是一个明智的生产目标，特别是如果您根据当前的安装指导来调整 API、worker 和数据库的大小。

## 后续步骤

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [CLI 参考](https://hindsight.vectorize.io/sdks/cli)
- [安装指南](https://hindsight.vectorize.io/sdks/developer/installation)
- [快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart)
- [文档首页](https://hindsight.vectorize.io)
