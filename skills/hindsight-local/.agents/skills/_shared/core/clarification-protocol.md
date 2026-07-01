# Clarification Protocol

When requirements are ambiguous, "assuming and proceeding" usually leads in the wrong direction.
Follow this protocol to secure clear requirements before execution.

> **Core Principle**: "Agents learn when to ask for help rather than blindly attempting every task" - Ask early, ask often.

---

## Uncertainty Level Definitions

| Level | State | Action | Example |
|-------|-------|--------|---------|
| **LOW** | Clear | Apply defaults and proceed, record assumptions | "Create a TODO app" |
| **MEDIUM** | Partially ambiguous | Present 2-3 options + request user selection | "Create a user management system" |
| **HIGH** | Very ambiguous | **Cannot proceed**, must ask questions | "Create a good app" |

---

## Uncertainty Triggers

Automatically classify as MEDIUM/HIGH level in the following situations:

### HIGH (Must Ask)
- [ ] Business logic decisions needed (pricing policy, approval workflow, etc.)
- [ ] Security/authentication decisions (OAuth provider, permission model, etc.)
- [ ] Possible conflict with existing code
- [ ] Requirements are subjective ("good", "fast", "pretty")
- [ ] Scope feels unlimited

### MEDIUM (Present Options)
- [ ] 2+ technology stack choices possible
- [ ] Trade-offs exist for implementation approach
- [ ] Multiple features with unclear priority
- [ ] External API/service selection needed

---

## Escalation Templates

### LOW → Proceed (Assumed)
```
Assumptions applied:
- JWT authentication included
- PostgreSQL database
- REST API
- MVP scope (CRUD only)

Proceeding with these defaults. Override if needed.
```

### MEDIUM → Request Selection (Options)
```
Uncertainty detected: {specific issue}

Option A: {approach}
  Pros: {benefits}
  Cons: {drawbacks}
  Effort: {low/medium/high}

Option B: {approach}
  Pros: {benefits}
  Cons: {drawbacks}
  Effort: {low/medium/high}

Option C: {approach}
  Pros: {benefits}
  Cons: {drawbacks}
  Effort: {low/medium/high}

Which approach do you prefer? (A/B/C)
```

### HIGH → Blocked
```
Cannot proceed: Requirements too ambiguous

Specific uncertainty: {what is unclear}

Questions needed:
1. {specific question}
2. {specific question}
3. {specific question}

Impact of proceeding blindly: {what could go wrong}

Status: BLOCKED (awaiting clarification)
```

---

## Required Verification Items

If any of the items below are unclear, **do not assume**; explicitly record them.

### Common to All Agents
| Item | Verification Question | Default (if assumed) | Uncertainty |
|------|----------------------|---------------------|-------------|
| Target users | Who will use this service? | General web users | LOW |
| Core features | What are the 3 must-have features? | Infer from task description | MEDIUM |
| Tech stack | Are there specific framework constraints? | Project default stack | LOW |
| Authentication | Is login required? | JWT authentication included | MEDIUM |
| Scope | MVP or full-featured? | MVP | LOW |

### Backend Agent Additional Verification
| Item | Verification Question | Default | Uncertainty |
|------|----------------------|---------|-------------|
| DB selection | PostgreSQL? MongoDB? SQLite? | PostgreSQL | MEDIUM |
| API style | REST? GraphQL? gRPC? | REST | MEDIUM |
| Auth method | JWT? Session? OAuth? | JWT (access + refresh) | HIGH |
| File upload | Needed? Size limit? | Not needed | LOW |
| Deployment environment | Serverless? Container? VM? | Container | MEDIUM |

### Frontend Agent Additional Verification
| Item | Verification Question | Default | Uncertainty |
|------|----------------------|---------|-------------|
| SSR/CSR | Server-side rendering needed? | Next.js App Router (SSR) | MEDIUM |
| Dark mode | Support needed? | Supported | LOW |
| Internationalization | Multi-language support? | Not needed | LOW |
| Existing design system | UI library to use? | shadcn/ui | MEDIUM |
| State management | Context? Redux? Zustand? | Zustand | MEDIUM |

### Mobile Agent Additional Verification
| Item | Verification Question | Default | Uncertainty |
|------|----------------------|---------|-------------|
| Platform | iOS only? Android only? Both? | Both | MEDIUM |
| Offline | Offline support needed? | Not needed | LOW |
| Push notifications | Needed? | Not needed | LOW |
| Minimum OS | iOS/Android minimum versions? | iOS 14+, Android API 24+ | LOW |
| Architecture | MVC? MVVM? Clean? | MVVM | MEDIUM |

---

## Detailed Response by Ambiguity Level

### Level 1 (LOW): Slightly ambiguous (core is clear, details lacking)
Example: "Create a TODO app"

**Response**: Apply defaults and record assumption list in result
```
Assumptions:
- JWT authentication included
- PostgreSQL database
- REST API
- MVP scope (CRUD only)
```

### Level 2 (MEDIUM): Considerably ambiguous (core features unclear)
Example: "Create a user management system"

**Response**: Narrow scope to 3 core features, specify and proceed
```
Interpreted scope (3 core features):
1. User registration + login (JWT)
2. Profile management (view/edit)
3. Admin user list (admin role only)

NOT included (would need separate task):
- Role-based access control (beyond admin/user)
- Social login (OAuth)
- Email verification
```

### Level 3 (HIGH): Very ambiguous (direction itself unclear)
Example: "Create a good app", "Improve this"

**Response**: Do not proceed, record clarification request in result
```
Cannot proceed: Requirements too ambiguous

Questions needed:
1. What is the app's primary purpose?
2. Who are the target users?
3. What are the 3 must-have features?
4. Are there existing designs or wireframes?

Status: blocked (awaiting clarification)
```

---

## PM Agent Only: Requirements Specification Framework

PM Agent uses the framework below to specify ambiguous requests:

```
=== Requirements Specification ===

Original request: "{user's original text}"

1. Core goal: {define in one sentence}
2. User stories:
   - "As a {user}, I want to {action} so that {benefit}"
   - (minimum 3)
3. Feature scope:
   - Must-have: {list}
   - Nice-to-have: {list}
   - Out-of-scope: {list}
4. Technical constraints:
   - {existing code / stack / compatibility}
5. Success criteria:
   - {measurable conditions}
```

---

## Application in Subagent Mode

CLI subagents cannot ask users directly.
Therefore:

1. **Level 1**: Apply defaults + record assumptions → Proceed
2. **Level 2**: Narrow and interpret scope + specify → Proceed
3. **Level 3**: `Status: blocked` + question list → Do not proceed

When Orchestrator receives Level 3 result, it relays questions to user
and re-runs that agent after receiving answers.
