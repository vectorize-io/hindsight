---
title: "在 Hindsight 的新列表视图中使用心理模型标签"
authors: [benfrank241]
date: 2026-04-28T14:00:00Z
tags: [mental-models, tags, control-plane, guide]
description: "使用 Hindsight 心理模型标签与新列表视图、服务器端标签建议以及更安全的刷新和反射工作流过滤。"
image: /img/guides/guide-use-mental-model-tags-in-hindsight-list-view.png
hide_table_of_contents: true
---

![在 Hindsight 的新列表视图中使用心理模型标签](/img/guides/guide-use-mental-model-tags-in-hindsight-list-view.png)

如果您想**在 Hindsight 的新列表视图中使用心理模型标签**，重要的变化是控制平面现在将心理模型视为真正的工作文档。页面默认为分割窗格列表，标签建议现在来自实际的心理模型标签源，而不是让您手动记住标签。在调整工作流时，请保持打开 [心理模型 API 文档](https://hindsight.vectorize.io/sdks/api/mental-models)、[反射文档](https://hindsight.vectorize.io/sdks/api/reflect)、[观察文档](https://hindsight.vectorize.io/sdks/developer/observations) 和 [文档首页](https://hindsight.vectorize.io)。

<!-- truncate -->

## 快速答案

- 心理模型现在在默认列表视图中打开，旧的卡片仪表板仍作为辅助视图可用。
- 标签建议是从新的心理模型标签源获取的，这意味着过滤现在基于您的模型实际使用的标签。
- 心理模型标签影响刷新输入和反射可见性，因此过滤不仅仅是装饰性的。

## 新列表视图改变了什么

列表视图是更好的默认值，因为心理模型是长期的操作文档，而不仅仅是仪表板卡片。您可以在一个窗格中扫描名称、源查询和刷新状态，然后在另一个窗格中检查内容。

这听起来很小，但它修复了一个真正的工作流问题。当您维护有多个心理模型的库时，卡片网格对浏览很好，而列表对实际维护更好。此更新保持两者，但它使维护视图成为您首先登陆的第一件事。

## 使用真实建议按标签过滤

新的标签建议路径很重要，因为它直接查询心理模型标签源。在实践中，这意味着 UI 可以建议来自心理模型的标签，而不是来自一般记忆。

如果您想检查基础端点形状，其想法是：

```bash
curl "$BASE_URL/v1/default/banks/$BANK_ID/tags?source=mental_models"
```

当库有许多普通的内存标签，但只有一小部分标签附加到心理模型时，这正是您想要的区别。

## 记住心理模型标签真正做什么

这是绊倒人们的部分。心理模型标签不仅仅是浏览标签。正如 [心理模型 API 文档](https://hindsight.vectorize.io/sdks/api/mental-models) 所解释的那样，标签缩小了模型在刷新期间可以读取哪些记忆，它们也影响 [反射](https://hindsight.vectorize.io/sdks/api/reflect) 期间哪些心理模型可见。

因此，如果您用 `user:alice` 标记心理模型，刷新将仅读取也带有该必需标签的记忆。当您想要范围模型时这很好，但这也意味着过度标记可以使模型看起来空或陈旧，如果基础记忆从未反向填充。

## 有意使用任何与全部

控制平面现在支持具有匹配模式的标签过滤。在实践中，这为您提供了两个有用的习惯：

- 在浏览和尝试快速找到相关心理模型时使用**任何**
- 当您调试精确范围并想仅查看匹配整个标签集的模型时，使用**全部**

如果过滤列表突然看起来太小，第一个要尝试的是从**全部**切换回**任何**。第二件事是检查心理模型本身是否具有比您期望读取的记忆更严格的标签。

## 排除空白或误导结果

一些模式经常出现：

- **模型存在，但内容为空。** 通常心理模型标签比可用的记忆更严格。
- **您想要的标签芯片从不出现。** 通常没有心理模型携带该标签，即使普通记忆也是如此。
- **反射似乎缺少模型。** 通常反射调用使用不与模型自己的标签重叠的标签。

如果有疑问，请通过 [心理模型 API 文档](https://hindsight.vectorize.io/sdks/api/mental-models) 和 [观察指南](https://hindsight.vectorize.io/sdks/developer/observations) 退一步。大多数心理模型混淆实际上是标签范围混淆。

## 常见问题

### 仪表板视图消失了吗？

不。仪表板仍然在那里。此更新使列表视图成为默认值，因为它对日常维护更好。

### 为什么心理模型标签建议需要单独的源？

因为内存标签和心理模型标签解决不同的浏览问题。当建议来自您实际过滤的心理模型表时，建议会更有用。

### 标签可以使心理模型从反射中消失吗？

是的。反射可见性也由标签过滤，因此请求标签和模型标签之间的不匹配可以隐藏模型。

## 后续步骤

- [Hindsight Cloud](https://hindsight.vectorize.io)
- [心理模型 API 文档](https://hindsight.vectorize.io/sdks/api/mental-models)
- [反射文档](https://hindsight.vectorize.io/sdks/api/reflect)
- [观察指南](https://hindsight.vectorize.io/sdks/developer/observations)
- [文档首页](https://hindsight.vectorize.io)
