---
title: "Enterprise Features in Hindsight Cloud"
authors: [benfrank241]
slug: "2026/09/03/hindsight-cloud-enterprise-features"
date: 2026-09-03T20:00
tags: [hindsight-cloud, enterprise, security, sso, mfa, audit, memory-defense, agent-memory]
description: "Every enterprise capability in Hindsight Cloud, and the failure each one prevents: single sign-on, MFA enforcement, role-based access, scoped API keys, audit logging, Memory Defense, webhooks, and bring-your-own-cloud or on-premise deployment."
image: /img/blog/hindsight-cloud-enterprise-features.png
hide_table_of_contents: true
---

![Enterprise features in Hindsight Cloud: single sign-on, MFA enforcement, role-based access, scoped API keys, audit logging, Memory Defense, and bring-your-own-cloud](/img/blog/hindsight-cloud-enterprise-features.png)

Most enterprise feature lists are indistinguishable from each other. SSO, audit logs, roles, private deployment — everyone ships them, everyone lists them, and the list tells you nothing about whether any of it actually holds.

So this is the list, but each entry says what it prevents. A control that can be routed around isn't a control, and a few of the ones below exist specifically because the obvious implementation doesn't hold.

<!-- truncate -->

## At a glance

| Capability | What it prevents | Availability |
|---|---|---|
| [**Single sign-on**](#single-sign-on) | Accounts that outlive someone's employment | Enterprise |
| [**MFA enforcement**](#multi-factor-authentication) | A password being the only thing in the way | Enterprise |
| [**Idle session timeout**](#idle-session-timeout) | An unattended session staying open | All plans |
| [**Role-based access**](#role-based-access) | Everyone having permission to do everything | All plans |
| [**API key controls**](#api-key-controls) | One leaked key exposing every bank you own | All plans |
| [**Scoped child keys**](#scoped-child-keys) | A tenant-isolation bug becoming a data leak | All plans |
| [**Audit logging**](#audit-logging) | Not being able to answer "who read this?" | Enterprise |
| [**Memory Defense**](#memory-defense) | Secrets being retained and indexed forever | All plans |
| [**Memory Defense Enterprise**](#memory-defense-enterprise) | Injections and payloads that poison recall | Enterprise |
| [**Webhooks**](#webhooks) | Finding out about an event by polling for it | Enterprise |
| [**Bring your own cloud**](#bring-your-own-cloud) | Data leaving your account, or your hardware | Enterprise |

## Single sign-on

*Enterprise.*

The reason SSO matters isn't sign-in convenience. It's **offboarding**. When someone leaves, you want one revocation in one system, not a hunt through every vendor for a lingering account.

Hindsight Cloud federates to your identity provider over **OIDC or SAML**, so Okta, Entra, Google Workspace, and anything standards-compliant will work. You claim your email domains and verify them by DNS; once a domain is verified and the config is active, anyone signing in with an address at that domain is routed to your IdP automatically. Members are provisioned on first sign-in and show up in the team list marked as SSO-provisioned, so there's no pre-creation step to maintain and no shadow list to clean up.

The part that makes it hold: **once SSO is active for a domain, password sign-in and password reset are refused for that domain.** An IdP you can sidestep by clicking "forgot password" isn't enforcing anything, and that's the default failure mode of bolt-on SSO. Here the password path is closed, not merely discouraged.

Client secrets are encrypted at rest and never returned by the API, only ever as a `has_client_secret` boolean.

## Multi-factor authentication

*Enterprise.*

An owner can require MFA across the organization. Two factors are supported: an **authenticator app** (TOTP, so 1Password, Authy, Google Authenticator, or anything TOTP-compatible) and an **emailed one-time code**.

The distinction worth understanding: enforcement checks that **the current session was authenticated with a factor**, not merely that the account has one enrolled. Those are different guarantees. An account with a dormant TOTP secret attached satisfies the weak version and nothing else. Every request is gated on the session itself having cleared MFA.

Individuals can enroll voluntarily from their account security page even when the organization doesn't require it.

MFA enforcement and SSO are alternatives rather than a stack. If sign-in routes through your IdP, MFA is enforced there under your own policy, so the two aren't switched on together for one organization.

## Idle session timeout

*All plans.* Thirty minutes of inactivity ends the session and requires a fresh sign-in. This is now on for every organization, not just Enterprise.

Unglamorous, and the one control on this list that guards against a threat with no technical component at all: an unlocked laptop.

## Role-based access

*All plans.* Three roles, deliberately few:

| Role | Permissions |
|---|---|
| **Owner** | Full access, including organization deletion |
| **Admin** | Manage members and API keys |
| **Member** | View-only access, and can use API keys |

Note where the line falls. A Member can *use* API keys but can't *manage* them, which is the split that matters in practice: your engineers need to build against memory without being able to mint credentials that outlive their access.

## API key controls

*All plans.* Keys are how your application actually reaches memory, so a leaked key is the realistic incident, not a stolen console password.

Every key can carry an **expiry** — a set number of hours or days, or never — and be **scoped to specific banks**, so it reaches only the memory it needs rather than everything the organization owns. Keys are **shown exactly once**, at creation.

Scoping is the one people skip, and it's the one that bounds a bad day. An unscoped key that leaks exposes every bank you have. A key scoped to one bank exposes one bank.

## Scoped child keys

*All plans.*

This one changes how you build a multi-tenant product, so it's worth more than a bullet.

A key can be granted permission to **programmatically create, list, and revoke bank-scoped child keys**. Your backend holds a single parent key and mints one child key per customer, each reaching only that customer's bank.

Consider the alternative. Without it, a multi-tenant app holds one broad credential and enforces tenant separation in its own code — a `WHERE tenant_id = ?` in every path that touches memory. That works until one path forgets, and the failure mode of forgetting is serving one customer's memories to another.

Minting scoped keys moves the boundary out of your code and into the credential. A child key that can only see one bank cannot reach another, no matter what the calling code does wrong. Isolation stops being a thing you remember to do correctly and becomes a property of the token.

Revocation inherits the same shape: kill one child key and that tenant's access ends, with nothing else disturbed.

## Audit logging

*Enterprise.*

A queryable trail of what happened in a bank — retain, recall, reflect, and other audited operations — with filters by action and severity, and request volume charted over time.

It's **opt-in per bank** rather than deployment-wide, which sounds like a limitation and isn't. Audit at scale produces an enormous amount of data, most of it about banks nobody will ever ask questions about. Enabling it per bank means the ones holding regulated or customer data get a complete trail, while the bank someone spun up to test a prompt doesn't bury it in noise.

## Memory Defense

*All plans.*

Memory Defense inspects content during retain and decides what to do with it **before anything is written**.

Timing is the whole design. Filtering at recall would be too late: by then the material is stored, embedded, and indexed, and anything with direct database or API access reads it regardless of what recall chooses to return.

Every bank gets a master switch, a default action for content matching no specific rule, and **secret masking** — detected API keys and tokens are replaced with markers like `[REDACTED:github_token]`, so the surrounding context survives while the credential doesn't.

That last property matters more than it first appears. Redaction rather than rejection means the memory is still useful. A conversation that happened to contain a token still gets retained as a memory; it just doesn't carry the token.

## Memory Defense Enterprise

*Enterprise.*

Additional detectors, each enabled and tuned on its own:

| Detector | What it catches |
|---|---|
| **Prompt injection guard** | Attempts to plant instructions in memory — "ignore previous", system overrides |
| **Size anomaly** | Oversized payloads. Default threshold 64 KB |
| **Protected document tags** | Re-submissions that would strip or replace a protected tag |
| **Broad pattern catalog** | 220+ secret-detection patterns on top of the basic set |
| **Base64 decoding** | Decodes base64 blobs and re-scans them, catching what a plain pattern match misses |
| **LLM screening** | Natural-language disclosures no pattern will find, like "the password is ..." |

Each detector carries its own **action** and **minimum severity**, which is what makes the feature survive contact with production. A scanner that can only block gets switched off the first time it rejects something legitimate. Being able to block on high confidence and merely flag on low confidence is the difference between a control that stays on and one that doesn't.

Entitlement is enforced **server-side**, not just rendered in the console, so a policy can't be set through the API that the UI wouldn't allow.

Why these particular detectors: memory is durable and semantically indexed, and that changes what a bad write costs. An injection that reaches a chat window affects one response. An injection that reaches memory is recalled on future turns, in future sessions, for as long as it matches. A secret retained once stays retrievable after you've rotated it. An oversized payload crowds the candidate set and pushes relevant memories out of results. These aren't generic content filters; they're aimed at the three things that go wrong specifically because the system remembers.

## Webhooks

*Enterprise.* Push memory-bank events to your own systems the moment they happen, rather than polling for them.

- **Real-time events** for completed retains and consolidations.
- **HMAC-signed payloads**, so your endpoint can verify a delivery actually came from us.
- **Asynchronous delivery with retries**, and a per-delivery history when you need to see what happened.
- **Memory Defense violation alerts**, which is the one that matters most here: a blocked secret or a caught injection can raise an event straight into your SIEM or incident tooling rather than sitting in a log nobody reads.

That last point is what turns Memory Defense from a filter into something your security team can actually monitor.

## Bring your own cloud

*Enterprise.*

Some requirements can't be met by any amount of application-level control, because the constraint isn't "who can read this" but "this data does not leave our account."

For those, Hindsight deploys **into your own cloud account** on AWS, GCP, or Azure.

**Private connectivity comes with it.** Traffic between your application and Hindsight stays on your cloud provider's private network instead of crossing the public internet, using whichever private-endpoint mechanism your provider offers.

**On-premise deployment is also available.** If the requirement is that data never leaves your own hardware, not merely your own cloud account, that's supported as an Enterprise deployment too.

## Frequently asked

**Does Hindsight Cloud support SAML, or only OIDC?**
Both. A configuration is set up as one or the other.

**Can we require MFA without using SSO?**
Yes. An organization can require MFA on Hindsight accounts without configuring an identity provider at all.

**If someone leaves, how do we cut off access?**
With SSO, deprovisioning in your IdP removes their route in; there's no separate Hindsight password still working. For programmatic access, revoke the key. If your application mints scoped child keys, revoking one ends that scope without touching any other.

**Does Memory Defense run on retain or on recall?**
On retain, before content is written.

**Can console entitlements be bypassed through the API?**
No. Entitlement is enforced server-side on the write path.

**We can't put data in a shared environment at all. Is that supported?**
Yes, either as a bring-your-own-cloud deployment into your own AWS, GCP, or Azure account, with private connectivity so traffic never crosses the public internet, or as an on-premise deployment if the data has to stay on your own hardware.

**Can we get events pushed to us instead of polling?**
Yes, through webhooks. Completed retains and consolidations raise events, payloads are HMAC-signed, delivery is asynchronous with retries and a per-delivery history, and Memory Defense violations can be routed to your SIEM.

## Learn more

- [What's New in Hindsight Cloud: June–August Updates](/blog/2026/09/01/hindsight-cloud-june-august-updates) — the most recent Cloud roundup
- [Per-User Memory for AI Products: Multi-Tenant Patterns](/blog/2026/08/04/per-user-multi-tenant-agent-memory) — the bank strategy behind scoped keys
- [Cross-Encoder Reranking: The Last Stage of Agent Memory Recall](/blog/2026/08/28/cross-encoder-reranking-agent-memory) — how recall decides what an agent sees
- [Knowledge Graphs vs. Vector Search for Agent Memory](/blog/2026/08/24/knowledge-graphs-vs-vector-search-agent-memory) — why retrieval runs several arms at once
