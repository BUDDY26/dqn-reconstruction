# Documentation Skill

**Purpose:** Generate and maintain docstrings, README content, architecture docs, and ADRs.

## When to Run
Say: "Document this", "Write a docstring for...", "Update the README", "Write an ADR for..."

## Scope

### Docstrings
- Use Google-style docstrings for Python
- Include: summary line, Args, Returns, Raises, Example (for public APIs)
- Do not add docstrings to trivial one-liner private functions

### README Updates
- Keep the README accurate with the current state of `src/`
- Update the feature list and usage examples when the public API changes
- Do not change the README structure without approval

### Architecture Docs (`docs/architecture.md`)
- Update when a new component is added or the data flow changes
- Include: component diagram description, data flow, key design decisions

### Architecture Decision Records (`docs/adr/`)
- Create one ADR per significant technical decision
- Copy `ADR-001-template.md` and fill in all sections
- Number ADRs sequentially: `ADR-002-*.md`, `ADR-003-*.md`, etc.
- ADRs are immutable once accepted — create a new ADR to supersede

## Output Format (ADR)

```
# ADR-NNN: <Title>

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Deprecated | Superseded by ADR-NNN

## Context
...

## Decision
...

## Consequences
...
```
