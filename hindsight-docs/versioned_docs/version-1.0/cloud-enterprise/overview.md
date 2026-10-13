---
title: Ways to run Hindsight
sidebar_label: Overview
description: Run open-source Hindsight yourself, use the managed Hindsight Cloud, or get a dedicated Hindsight Enterprise deployment.
---

# Ways to run Hindsight

Hindsight runs three ways. All of them use the same memory engine: the same retain, recall and reflect, the same mental models and knowledge pages, and the same API and SDKs. You can start on one and move to another later.

| | Self-hosted (open source) | Hindsight Cloud | Hindsight Enterprise |
|---|---|---|---|
| **What it is** | The open-source engine, run by you | Managed, self-serve, multi-tenant service | A single-tenant deployment, installed and run with Vectorize |
| **Where it runs** | Anywhere: your laptop, Docker, Kubernetes | Hindsight Cloud | Our cloud, your AWS, Google Cloud or Azure account, or your data center |
| **Who runs it** | You | Vectorize | Vectorize, or you with Vectorize's help on-premises |
| **Memory store** | PostgreSQL (or Oracle) | Managed by Vectorize | PostgreSQL, or [Hindsight Memlake](./enterprise.md#hindsight-memlake) on object storage |
| **Models** | Bring your own LLM and embeddings | Included | Included, or your own (Bedrock, Vertex AI, Azure OpenAI) |
| **Sign-in and access** | API keys you configure | Console, teams and API keys | Plus SSO, enforced MFA and private networking |
| **Pricing** | Free (MIT license) | Pay as you go, from prepaid credits | Contract |
| **Get started** | [Installation](/developer/installation) | [Sign up](https://ui.hindsight.vectorize.io/signup) | [Book a call](https://calendly.com/d/dvy5-v3f-jhg?utm_source=oss_docs&utm_campaign=enterprise_docs) |

## How to choose

- **Self-hosted** if you want full control, want to try Hindsight locally, or already run Postgres and your own models. These docs cover it: start with [Installation](/developer/installation).
- **[Hindsight Cloud](./hindsight-cloud.md)** if you want memory for your agents in minutes, with nothing to deploy and the models included.
- **[Hindsight Enterprise](./enterprise.md)** if you need your own deployment: single-tenant isolation, a region or cloud account of your choice, private networking, SSO and compliance features, contracted support, or memory at a scale where [Hindsight Memlake](./enterprise.md#hindsight-memlake) pays off.

Hindsight Cloud and Enterprise have their own documentation at [docs.hindsight.vectorize.io](https://docs.hindsight.vectorize.io).
