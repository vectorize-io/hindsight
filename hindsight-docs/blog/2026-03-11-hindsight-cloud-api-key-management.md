---
title: "What's New in Hindsight Cloud: Programmatic API Key Management"
authors: [benfrank241]
date: 2026-03-11T12:00
tags: [hindsight-cloud, release, api]
hide_table_of_contents: true
---

[Hindsight Cloud](https://ui.hindsight.vectorize.io) now supports programmatic API key management. API keys with the **Key Creator** capability can create, list, and revoke bank-scoped child keys via the API — no admin or UI access required.

<!-- truncate -->

## What You Can Do

- **Create child keys** — provision short-lived, least-privilege keys scoped to specific banks
- **Bank scope enforcement** — child keys can only access banks within the parent key's scope
- **Expiration constraints** — child keys cannot outlive their parent
- **Cascade revocation** — revoking a parent key automatically revokes all of its children
- **Immutable children** — programmatically created keys cannot have their bank scope edited; revoke and recreate instead
- **Audit trail** — all key creation, revocation, and scope changes are logged with actor identity

## Why This Matters

If you're running a multi-tenant setup — one memory bank per customer, per agent, or per environment — you no longer need to manage keys through the dashboard. Your application can provision scoped keys on the fly, rotate them on a schedule, and revoke them instantly when access should end.

Combined with [bank-scoped API keys](/blog/2026/03/09/hindsight-document-upload#what-else-is-new) (released March 9), this gives you a complete least-privilege key hierarchy: a parent key with Key Creator capability manages child keys that are each locked to specific banks.

## Get Started

Programmatic API key management is available now in [Hindsight Cloud](https://ui.hindsight.vectorize.io). Create a key with the Key Creator capability to start provisioning child keys via the API.
