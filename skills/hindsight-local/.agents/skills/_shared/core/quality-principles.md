# Oh-My-AG Quality Principles

4 principles to follow across all development phases.

---

## 1. Think Before Coding

**Core**: Surface assumptions, ask when uncertain, present alternatives.

### Application
- List assumptions explicitly
- Present 2+ alternatives for architecture decisions
- Stop and clarify when confused
- Push back on unclear requirements

### Anti-patterns
- Starting without understanding
- Silently choosing one interpretation
- Hiding confusion

---

## 2. Simplicity First

**Core**: Minimum code that solves the problem. Nothing speculative.

### Application
- Implement exactly what's specified
- No "while I'm here" additions
- Skip abstractions for single-use code
- Rewrite if 200 lines could be 50

### Anti-patterns
- Feature creep
- "Might need later" code
- Unrequested flexibility

---

## 3. Surgical Changes

**Core**: Touch only what you must. Clean up only your own mess.

### Application
- Modify only task-related files
- Match existing style
- Remove only orphans YOUR changes created
- Flag unrelated issues without fixing

### Anti-patterns
- Drive-by refactoring
- Improving adjacent code
- Cleaning pre-existing debt

---

## 4. Verify Until Done

**Core**: Define success criteria. Loop until verified.

### Application
- Transform tasks into verifiable goals
- "Add validation" → "Write failing tests, then pass them"
- "Fix bug" → "Create reproducing test, then pass it"
- Run automated verification at each checkpoint

### Anti-patterns
- Declaring victory without testing
- Assuming correctness
- Skipping verification steps

---

## Quick Reference

| Principle | One-liner | Question to Ask |
|-----------|-----------|-----------------|
| Think Before Coding | "What could go wrong?" | "Did I list my assumptions?" |
| Simplicity First | "Do we need this?" | "Would a senior call this overcomplicated?" |
| Surgical Changes | "Is this in scope?" | "Does every line trace to the request?" |
| Verify Until Done | "How do I prove it works?" | "What's my success criteria?" |
