---
title: "安全地减少 Hindsight 合并内存扇出"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [consolidation, memory, operations, guide]
description: "通过调整回忆预算、源事实限制和大型库上的 FlashRank 内存设置，减少 Hindsight 合并内存扇出。"
image: /img/guides/guide-reduce-hindsight-consolidation-memory-fan-out.png
hide_table_of_contents: true
---

![安全地减少 Hindsight 合并内存扇出](/img/guides/guide-reduce-hindsight-consolidation-memory-fan-out.png)

如果您需要**减少 Hindsight 合并内存扇出**，最近的默认值是一个真正的改进。合并过去能够在内部回忆期间放大内存使用，特别是在大型库上，其中源事实补水和重新排列器行为可能会使 RSS 保持比预期更高。新默认值使该路径更加受限。在调整时，请保持打开 [配置指南](https://hindsight.vectorize.io/sdks/developer/configuration)、[观察指南](https://hindsight.vectorize.io/sdks/developer/observations)、[安装指南](https://hindsight.vectorize.io/sdks/developer/installation) 和 [文档首页](https://hindsight.vectorize.io)。

<!-- truncate -->

## 快速答案

- 合并回忆现在默认为低预算，这减少了每个回忆臂尝试提取的候选行数。
- 合并内的源事实现在默认受到限制，而不是保持无限制。
- FlashRank 现在默认为受限的 CPU 内存竞技场设置，这有助于防止合并工作完成后 RSS 单调增长。

## 为什么合并过去会扇出

合并不仅仅是一个简单的合并通过。它可以触发内部回忆工作，以便系统可以找到相关的观察、补水源事实并决定更新或组合什么。在大型库上，如果回忆预算很宽、源事实实际上无限制且重新排列器运行时保持内存竞技场热备，这可能会很快变得昂贵。

这是此更新所解决的模式。它不会删除合并。它缩小了昂贵的部分，以便大型库保持更可预测。

## 首先使用新的受限默认值

新的基线故意保守：

```bash
export HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET=low
export HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096
export HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA=false
```

这些默认值减少了候选扇出，限制了在提示中提取的源证据，并阻止 ONNX Runtime 在合并批次完成后保持不断增长的 CPU 竞技场。

## 仅当您有理由时才调整

如果您超越默认值，请有目的地进行。

- 仅当低回忆明确缺少有用的相关观察时，才提高 `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET`。
- 仅当 LLM 需要更多支持证据来进行稳定更新时，才提高 `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS`。
- 如果您想在吞吐量与峰值内存压力之间进行权衡，请查看 [配置指南](https://hindsight.vectorize.io/sdks/developer/configuration) 中的 `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND` 和 `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE`。

关键是默认情况下保持昂贵的路径狭窄，然后如果库实际需要，一次只扩大一个杠杆。

## 也检查其余部分的部署形状

合并调整仅解决合并。如果 RSS 看起来仍然不好，请根据更广泛的部署进行比较：

- 您是否在运行完整映像而不是精简版？
- worker 是否与同一小主机上的 API 并置？
- PostgreSQL 是否与相同的内存信封共享？
- 当外部重新排列器会更好地适应时，您是否在使用本地重新排列？

这就是为什么 [安装指南](https://hindsight.vectorize.io/sdks/developer/installation) 和 [服务指南](https://hindsight.vectorize.io/sdks/developer/services) 在这里仍然重要。合并扇出是一个贡献者，而不是整个足迹故事。

## 一个简单的操作手册

一个理智的生产手册看起来像这样：

1. 从新的默认值开始。
2. 在大型合并轮次期间观察 RSS。
3. 一次只调整一个旋钮。
4. 在相同的库形状上重新测试。
5. 保持说明哪个变化实际上使针移动。

这个纪律很重要，因为当多个变量同时变化时，内存问题常常感觉神秘。

## 常见问题

### 低回忆预算会损害正常用户回忆质量吗？

不。此设置特定于合并内的内部回忆通过，而不是用户直接调用的一般回忆路径。

### 为什么将源事实限制为 4096 个令牌？

因为无限制的源事实补水是大型库上最糟糕的内存放大器之一。一个上限使提示成本变得更容易预测。

### 我应该将 FlashRank CPU 内存竞技场重新打开吗？

通常不。除非您已测量到真正的需要并且乐意将受限的 RSS 换成不同的分配模式，否则请将其关闭。

## 后续步骤

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [配置指南](https://hindsight.vectorize.io/sdks/developer/configuration)
- [观察指南](https://hindsight.vectorize.io/sdks/developer/observations)
- [安装指南](https://hindsight.vectorize.io/sdks/developer/installation)
- [文档首页](https://hindsight.vectorize.io)
