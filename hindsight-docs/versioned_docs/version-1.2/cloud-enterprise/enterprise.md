---
title: Hindsight Enterprise
description: Hindsight Enterprise is a single-tenant Hindsight deployment, in our cloud, your cloud or your data center, with Hindsight Memlake for memory at scale.
---

# Hindsight Enterprise

Hindsight Enterprise is Hindsight for organizations that need their own deployment. Instead of sharing the multi-tenant Hindsight Cloud, you get a **single-tenant** installation that Vectorize sets up and runs with you. It includes everything in Hindsight Cloud, plus the security, compliance, networking and scale features below.

**[Book a call](https://calendly.com/d/dvy5-v3f-jhg?utm_source=oss_docs&utm_campaign=enterprise_docs)** · **[Hindsight Enterprise docs](https://docs.hindsight.vectorize.io/enterprise)**

## Deployment options

| Option | Where it runs | Who runs it |
|---|---|---|
| **Hindsight Cloud with Enterprise features** | Hindsight Cloud (multi-tenant) | Vectorize |
| **Private cloud** | A dedicated deployment in Vectorize's cloud account, in the region you choose | Vectorize |
| **Bring your own cloud (BYOC)** | Your AWS, Google Cloud or Azure account | Vectorize, through access you grant |
| **On-premises** | Your Kubernetes infrastructure, including air-gapped | You, with Vectorize's help |

Private cloud and BYOC can be reached over private networking (AWS PrivateLink, Google Cloud Private Service Connect, Azure Private Link), so traffic never crosses the public internet. See [Choosing a deployment option](https://docs.hindsight.vectorize.io/enterprise/deployment/overview) for a full comparison, including who is responsible for what.

## What Enterprise adds

- **Single-tenant deployment:** its own database, storage, compute and identity service.
- **[Hindsight Memlake](#hindsight-memlake):** memory on object storage, for large and growing banks.
- **Single sign-on and MFA:** Okta, Microsoft Entra ID, Google Workspace or any OIDC or SAML provider.
- **Memory Defense:** detect and redact secrets and sensitive data, and block prompt injection before it is stored.
- **Data privacy (GDPR):** erase, retain and export a person's data across banks.
- **Audit logs, webhooks and SIEM:** a record of who did what, streamed to your own systems.
- **Enterprise support:** a ticket portal, a shared Slack channel and contracted response times.
- **Invoiced billing:** annual or contract billing instead of card payments.

## Hindsight Memlake

Hindsight Memlake is the Enterprise storage engine for agent memory. Open-source Hindsight stores memories in PostgreSQL, which works well for most banks. But agent memory only grows, and as banks reach hundreds of thousands of memories, a relational database gets slower and more expensive to keep fast.

Memlake keeps memories in **object storage** (S3, Google Cloud Storage, Azure Blob Storage, or an S3-compatible store on-premises), with one bucket per organization:

- **Built for scale.** In our tests, 100 million memories were served at under 100 ms p99 recall, on four servers.
- **Fast and steady.** Recall time barely moves as a bank grows, because each bank has its own index, kept warm in a cache on the servers.
- **Lower cost as you grow.** Memories live in object storage, the cheapest durable storage there is, and servers hold only a cache, so you add servers for traffic, not for data.
- **Durable.** A write is in object storage before it is acknowledged, and continuously backed up to a separate bucket.
- **Nothing changes for your agents.** Same API, SDKs and features; banks, settings and background jobs stay in PostgreSQL.

If your banks stay under roughly 50,000 memories each, PostgreSQL is a good fit and Memlake is not needed. Memlake is available on private cloud, BYOC and on-premises deployments.

Read more: [Hindsight Memlake overview](https://docs.hindsight.vectorize.io/enterprise/memlake), [architecture](https://docs.hindsight.vectorize.io/enterprise/memlake/architecture), [performance and cost](https://docs.hindsight.vectorize.io/enterprise/memlake/performance) and [tradeoffs](https://docs.hindsight.vectorize.io/enterprise/memlake/tradeoffs).

## Learn more

- [Hindsight Enterprise documentation](https://docs.hindsight.vectorize.io/enterprise)
- [Ways to run Hindsight](./overview.md): self-hosted, Cloud and Enterprise compared
- [Book a call](https://calendly.com/d/dvy5-v3f-jhg?utm_source=oss_docs&utm_campaign=enterprise_docs) with the Hindsight team
