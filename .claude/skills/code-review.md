# Code Review Skill

**Purpose:** Structured code review with severity-labeled findings.

## When to Run
Say: "Review this file" or "Review [filename]"

## Severity Labels

| Label | Meaning |
|-------|---------|
| `[CRITICAL]` | Bug, security flaw, or data loss risk — must fix before merge |
| `[HIGH]` | Significant correctness or performance issue |
| `[MEDIUM]` | Code quality, maintainability, or test coverage gap |
| `[LOW]` | Style, naming, or minor clarity improvement |
| `[INFO]` | Observation with no required action |

## Review Checklist

- [ ] Correctness — does the code do what it claims?
- [ ] Edge cases — are boundary conditions handled?
- [ ] Error handling — are failures surfaced appropriately?
- [ ] Test coverage — are critical paths tested?
- [ ] Documentation — are public interfaces documented?
- [ ] Security — are inputs validated at system boundaries?
- [ ] Performance — are there obvious inefficiencies?
- [ ] Naming — are identifiers clear and consistent?

## Output Format

```
## Code Review: <filename>

### [CRITICAL] <title>
<location> — <description> — <recommended fix>

### [HIGH] <title>
...

### Summary
<N> critical, <N> high, <N> medium, <N> low findings.
Recommendation: [Approve / Approve with changes / Revise and re-review]
```
