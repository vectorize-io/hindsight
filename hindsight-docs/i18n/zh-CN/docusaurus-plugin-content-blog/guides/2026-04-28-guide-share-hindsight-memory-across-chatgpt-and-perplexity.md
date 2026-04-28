---
title: "在 ChatGPT 和 Perplexity 之间共享 Hindsight 内存"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [shared-memory, chatgpt, perplexity, guide]
description: "通过将两个连接器指向同一个库在 ChatGPT 和 Perplexity 之间共享 Hindsight 内存，然后测试实际工作中的跨工具回忆。"
image: /img/guides/guide-share-hindsight-memory-across-chatgpt-and-perplexity.png
hide_table_of_contents: true
---

![在 ChatGPT 和 Perplexity 之间共享 Hindsight 内存](/img/guides/guide-share-hindsight-memory-across-chatgpt-and-perplexity.png)

如果您想**在 ChatGPT 和 Perplexity 之间共享 Hindsight 内存**，关键的决定不是连接器本身。这是库的边界。两个工具已经可以与 Hindsight 对话，但只有当它们故意写入并从同一个内存库读取时，它们才感觉像是一个工作流。在设置共享路径时，使用 [ChatGPT 指南](https://hindsight.vectorize.io/sdks/integrations/chatgpt)、[Perplexity 指南](https://hindsight.vectorize.io/sdks/integrations/perplexity)、[MCP 服务器文档](https://hindsight.vectorize.io/sdks/developer/mcp-server) 和 [快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart) 作为技术参考。

<!-- truncate -->

## 快速答案

- 要共享内存，将两个连接器指向同一个 Hindsight 库，而不是让每个工具保持自己隔离的默认值。
- 当工作是真正共享的时，共享库效果最好，例如 Perplexity 中的研究和 ChatGPT 中的综合。
- 最快的证明是在一个工具中保留某些内容，然后向另一个工具提出一个依赖它的后续问题。

## 有意选择一个库

共享内存设置只有在库边界与工作相匹配时才有用。选择一个映射到真实项目、团队或正在进行的研究线程的库名称。

例如，如果两个工具都在帮助同一次发布，请使用一个库，例如：

```text
https://api.hindsight.vectorize.io/mcp/product-launch/
```

这可以保持上下文的具体性。如果将无关的工作倒入同一个共享库，跨工具回忆就会变得嘈杂，整个设置开始感觉不那么值得信任。

## 将两个连接器指向同一 MCP 路径

在两个工具都单独工作后，更新每个连接器，以便 URL 解析为同一个库。

例如，两个工具都可以针对：

```text
https://api.hindsight.vectorize.io/mcp/product-launch/
```

然后保持自定义指令一致。要求两个工具保留持久的发现、决定和约束，而不是每个随意的旁述。如果 ChatGPT 存储架构选择，而 Perplexity 存储源支持的研究，共享库就变得比任何一个工具都更有价值。

## 使用一个具体的工作流测试跨工具回忆

简单的验证流程就足够了：

1. 在 Perplexity 中，研究一个主题并要求 Hindsight 保留主要发现。
2. 开始一个新的 ChatGPT 聊天。
3. 要求 ChatGPT 回答取决于这些发现的规划问题。
4. 检查它是否回忆了早期的研究，而不是从零开始。

您也可以反转测试。在 ChatGPT 中存储一个决定，然后使用 Perplexity 从该决定状态继续研究。重要的部分不是第一次尝试的完美。这是证明两个工具都在从同一持久层读取。

## 使用护栏使共享库保持有用

共享库并不意味着一个草率的库。一些护栏会产生很大的差异：

- 每个项目或域保持一个库，而不是一个库用于您的整个数字生活
- 为产品、repos 和倡议使用一致的名称
- 定期在 Hindsight Cloud 中查看嘈杂的记忆
- 如果研究和个人偏好数据不应混合，则再次分割工作

如果您需要更深入的控制，下一层是 [保留 API](https://hindsight.vectorize.io/sdks/api/retain)、[回忆 API](https://hindsight.vectorize.io/sdks/api/recall) 和来自 [文档首页](https://hindsight.vectorize.io) 的库级别配置。共享内存在保持范围时最有用。

## ChatGPT 和 Perplexity 各帮助最多的地方

这个设置有效，因为工具是不同的。Perplexity 擅长网络支持的发现、源收集和迭代研究。ChatGPT 擅长综合、规划、起草和推理权衡。

共享的 Hindsight 库让每个工具为另一个工具留下持久的工作产品。这是真正的回报。您不是跨选项卡复制上下文，而是让内存层携带它。

## 常见问题

### 两个工具应该总是共享一个库吗？

不。只有在工作真正共享时才这样做。对于无关的项目或不同的隐私边界，独立库仍然是正确的答案。

### 我可以从 `/mcp/default/` 开始，稍后切换吗？

是的。重要的是一致性。一旦您决定工作流应该被共享，将两个连接器都指向同一个命名库路径。

### 我应该从每个工具存储什么？

Perplexity 应该通常保留研究发现和来源。ChatGPT 应该通常保留决定、约束、草稿和以后重要的推理结果。

## 后续步骤

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [ChatGPT 集成指南](https://hindsight.vectorize.io/sdks/integrations/chatgpt)
- [Perplexity 集成指南](https://hindsight.vectorize.io/sdks/integrations/perplexity)
- [MCP 服务器文档](https://hindsight.vectorize.io/sdks/developer/mcp-server)
- [快速入门指南](https://hindsight.vectorize.io/sdks/developer/quickstart)
