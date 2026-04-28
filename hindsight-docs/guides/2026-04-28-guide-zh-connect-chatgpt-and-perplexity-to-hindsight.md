---
title: "将 ChatGPT 和 Perplexity 连接到 Hindsight 内存"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [integrations, chatgpt, perplexity, guide]
description: "使用 MCP、OAuth 和自定义指令将 ChatGPT 和 Perplexity 连接到 Hindsight，使两个工具都能保留上下文而不是从零开始。"
image: /img/guides/guide-connect-chatgpt-and-perplexity-to-hindsight.png
hide_table_of_contents: true
---

![将 ChatGPT 和 Perplexity 连接到 Hindsight 内存](/img/guides/guide-connect-chatgpt-and-perplexity-to-hindsight.png)

如果您想**将 ChatGPT 和 Perplexity 连接到 Hindsight**，这两个工具的连接路径现在已清晰记录。新的官方集成指南展示了相同的核心模式：添加远程 MCP 连接器、在浏览器中完成 OAuth、然后给模型明确的指令来保留和回忆有用的上下文。在设置时，请保持打开[ChatGPT 集成指南](https://hindsight.vectorize.io/sdks/integrations/chatgpt)、[Perplexity 集成指南](https://hindsight.vectorize.io/sdks/integrations/perplexity)、[MCP 服务器文档](https://hindsight.vectorize.io/sdks/developer/mcp-server)和[文档首页](https://hindsight.vectorize.io)。

<!-- truncate -->

## 快速答案

- 两个集成都使用 Hindsight MCP 端点和基于浏览器的 OAuth，因此无需在模型界面中粘贴 API 密钥。
- ChatGPT 和 Perplexity 都需要明确的指令，如果您想要自动保留和回忆行为而不是手动工具使用。
- 最简单的起点是每个工具一个内存库，只有在您真正想要跨工具内存时才移到共享库。

## 文档中的变化

最大的改进不是新的传输方式，而是清晰度。Hindsight 现在为 [ChatGPT](https://hindsight.vectorize.io/sdks/integrations/chatgpt) 和 [Perplexity](https://hindsight.vectorize.io/sdks/integrations/perplexity) 提供专门的设置页面，其中包含实际的 MCP URL、OAuth 流程和示例自定义指令。

这很重要，因为如果连接器存在但模型从不调用它，这些集成就只完成了一半。新文档明确说明：首先连接连接器，然后教模型何时保留以及何时回忆。

## 首先连接 ChatGPT

在 ChatGPT 中，转到**设置**，然后转到**应用和连接器**，然后创建一个指向以下地址的连接器：

```text
https://api.hindsight.vectorize.io/mcp/default/
```

创建后，浏览器应该打开 Hindsight OAuth 批准流程。完成该步骤，然后添加自定义指令，告诉 ChatGPT 在每个响应后保留重要事实、决定和约束，并在回答前回忆相关的记忆。

ChatGPT 中的设置页面很容易验证。您应该能够打开聊天、选择连接器并要求它记住一个具体的事实。如果那个往返失败，请停止并在调整提示之前修复连接器。

## 其次连接 Perplexity

Perplexity 使用相同的基本模型，但有一个额外的限制因素：远程 MCP 连接器需要 **Perplexity Pro**。

在**设置**中使用相同的 MCP 服务器 URL，然后转到**连接器**，然后转到**自定义连接器**。完成 OAuth 批准流程，然后添加自定义指令，告诉 Perplexity 在每次搜索后保留发现、来源和搜索模式。

Perplexity 在将当前网络搜索与您在早期研究会话中已经学到的内容相结合时最强大。这是 Hindsight 添加的部分。

## 决定是否要独立或共享内存

有两个明智的起点：

1. **ChatGPT 和 Perplexity 的独立库**，这更简单并减少意外混合。
2. **共享库**，当两个工具在同一项目上工作并应该建立在相同的研究、决定和约束之上时。

如果您不确定，请从独立开始。一旦有具体原因，再转到共享设置。共享模式在 [MCP 服务器文档](https://hindsight.vectorize.io/sdks/developer/mcp-server) 和 [Perplexity 指南](https://hindsight.vectorize.io/sdks/integrations/perplexity)的后续工作流中有更直接的描述。

## 排除第一次连接的故障

最常见的错误很无聊，但也很容易修复：

- 连接器存在，但模型从不调用它。这通常是自定义指令问题。
- OAuth 成功，但工具调用返回空结果。这通常是因为库没有有用的保留内容。
- 一个工具有效，另一个无效。这通常意味着您配置了不同的 URL 或在浏览器中批准了不同的 Hindsight 帐户。
- Perplexity 完全隐藏该功能。这通常意味着帐户不在 Pro 计划上。

将设置视为三个独立的检查：添加了连接器、批准了 OAuth、在实际对话中验证了工具行为。

## 常见问题

### 我需要在 ChatGPT 或 Perplexity 内添加 API 密钥吗？

不。文档化的集成路径使用针对 Hindsight MCP 服务器的基于浏览器的 OAuth。

### 我应该立即启用自动保留吗？

通常是的，但将指令集中在持久的事实、决定和发现上。保留每一个琐碎的交换会使库更杂乱。

### 我应该首先连接哪个工具？

从您使用更频繁的工具开始。ChatGPT 通常是推理工作流的更好初始设置，而 Perplexity 是研究密集工作的强大第二步。

## 后续步骤

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [ChatGPT 集成指南](https://hindsight.vectorize.io/sdks/integrations/chatgpt)
- [Perplexity 集成指南](https://hindsight.vectorize.io/sdks/integrations/perplexity)
- [MCP 服务器文档](https://hindsight.vectorize.io/sdks/developer/mcp-server)
- [文档首页](https://hindsight.vectorize.io)
