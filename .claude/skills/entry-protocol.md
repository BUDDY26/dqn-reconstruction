# Entry Protocol

**Purpose:** Mandatory repository scan to be completed before any code modifications in a new session.

## When to Run
Run this skill at the start of every Claude Code session by saying: "Run the entry protocol"

## Protocol Steps

### Phase 1 — Read operating context
- Read `CLAUDE.md` in full
- Read `docs/architecture.md` if it exists
- Note the permission tier (Section 3) and active hooks (Section 7)

### Phase 2 — Scan repository structure
- List all files in `src/` and `tests/`
- Identify entry points named in CLAUDE.md Section 2
- Confirm `scripts/validate-structure.sh` is present

### Phase 3 — Check for outstanding issues
- Note any `TODO`, `FIXME`, or `STUB` comments in source files
- Note any unfilled `{{PLACEHOLDER}}` tokens in `.md` files
- Check git status for uncommitted changes

### Phase 4 — Summarize and propose
- Produce a one-paragraph system summary
- Propose a prioritized list of improvements
- Do not modify any files until the user approves

## Output Format

```
## Entry Protocol Summary
**Project:** <name>
**State:** <what exists, what is missing>
**Pending placeholders:** <count>
**Proposed next steps:**
1. ...
2. ...
```
