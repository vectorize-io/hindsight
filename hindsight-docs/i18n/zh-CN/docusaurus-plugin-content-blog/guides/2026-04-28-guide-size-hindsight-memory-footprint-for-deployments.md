---
title: "为实际部署调整 Hindsight 内存占用空间"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [deployment, hardware, memory, guide]
description: "使用完整和精简映像、workers、控制平面和 PostgreSQL 容量规划的新内存占用空间指导来调整 Hindsight 部署。"
image: /img/guides/guide-size-hindsight-memory-footprint-for-deployments.png
hide_table_of_contents: true
---

![为实际部署调整 Hindsight 内存占用空间](/img/guides/guide-size-hindsight-memory-footprint-for-deployments.png)

如果您正在尝试**为部署调整 Hindsight 的内存占用空间**，文档现在好得多，因为安装指南最终按组件列出了现实的 RAM 范围。这很有用，因为合适的盒子大小取决于一个简单的问题：您运行带有本地模型的完整映像，还是带有外部提供者的精简映像？在规划时，请保持打开 [安装指南](https://hindsight.vectorize.io/sdks/developer/installation)、[配置指南](https://hindsight.vectorize.io/sdks/developer/configuration)、[服务指南](https://hindsight.vectorize.io/sdks/developer/services) 和 [快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart)。

<!-- truncate -->

## 快速答案

- 完整 API 映像需要约 1.5 GB 最小和 2 GB 推荐，因为它加载本地嵌入和重新排列器模型。
- 精简 API 映像可以运行约 512 MB 最小和 1 GB 推荐，但仅当嵌入和重新排列被卸载时。
- 控制平面很轻，workers 镜像 API 映像占用空间，PostgreSQL 仍需要自己的余量。

## 从当前基线数字开始

新的安装指导提供了一个实际的基线：

| 组件 | 最小 RAM | 推荐 RAM |
|---|---:|---:|
| API, 完整映像 | 1.5 GB | 2 GB |
| API, 精简映像 | 512 MB | 1 GB |
| 控制平面 | 128 MB | 256 MB |
| Worker | 与 API 变体相同 | 与 API 变体相同 |
| PostgreSQL | 512 MB | 1 GB+ |

这些不是理论楼层值。它们是反映本地模型成本、运行时开销以及 PostgreSQL 和 workers 仍需要呼吸空间的规划数字。

## 在选择机器之前选择完整或精简

道路上最大的分支是您是否想要本地嵌入和重新排列打包到 API 进程中。

- 当您想要更简单的一体式部署并且可以负担额外的 RAM 时，选择**完整**。
- 当您想要较小的主机并且乐意从 [配置指南](https://hindsight.vectorize.io/sdks/developer/configuration) 中连接外部提供者时，选择**精简**。

这个决定通常比争论小 VM 家族更重要。完整映像购买便利性。精简映像购买了更小的占用空间。

## 使用实用的部署食谱

一些起点效果很好：

- **笔记本电脑或个人开发盒**：完整映像、嵌入式数据库、2 vCPU、2 GB 到 4 GB RAM。
- **小型云 VM**：精简映像、外部嵌入和重新排列器、1 GB 到 2 GB RAM 加一个单独的数据库。
- **更重的生产设置**：API 和 worker 分离、PostgreSQL 独立调整大小、如果回忆延迟重要，重新排列器离开盒子。

这就是为什么文档现在将 API、worker、UI 和数据库指导分开。Hindsight 是一个产品，但不是一个内存占用空间。

## 知道压力真正来自哪里

对于生产流量，重新排列器通常是第一个使主机感觉昂贵的东西。在仅 CPU 的盒子上，它可能成为主要的延迟和内存压力点。这就是为什么安装文档现在说 CPU 对开发和基本工作负载很好，但生产流量经常受益于 GPU 支持的重新排列或外部重新排列服务。

换句话说，如果部署感觉比预期更大，罪魁祸首通常是模型位置，而不是控制平面或简单的文档页面计数。

## 排除内存压力而不猜测

如果一个节点运行得很热，请按顺序完成堆栈：

1. 确认您是在完整还是精简映像上。
2. 检查 workers 是否与同一主机共享并加倍预期的模型占用空间。
3. 验证 PostgreSQL 在同一盒子上没有被饿死。
4. 查看 [配置指南](https://hindsight.vectorize.io/sdks/developer/configuration) 中的外部提供者设置。
5. 根据 [服务指南](https://hindsight.vectorize.io/sdks/developer/services) 比较部署形状（如果您拆分 API 和 worker 角色）。

新文档不会删除调整工作，但它确实使第一个估计变得更加不含糊。

## 常见问题

### 我可以在 1 GB RAM 中运行 Hindsight 吗？

是的，但通常只有使用精简映像和外部提供者。完整映像不是该信封的正确选择。

### Workers 需要单独的调整大小吗？

是的。Workers 加载与 API 映像变体相同的模型堆栈，因此应该像另一个 API 进程一样进行预算。

### 控制平面是昂贵的部分吗？

不。控制平面相对较轻。本地嵌入和重新排列主导占用空间。

## 后续步骤

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [安装指南](https://hindsight.vectorize.io/sdks/developer/installation)
- [配置指南](https://hindsight.vectorize.io/sdks/developer/configuration)
- [服务指南](https://hindsight.vectorize.io/sdks/developer/services)
- [快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart)
