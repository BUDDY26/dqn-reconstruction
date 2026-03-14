# Refactor Playbook

**Purpose:** Safe, proposal-first refactoring workflow. No code is changed until a written proposal is approved.

## When to Run
Say: "Refactor this" or "Refactor [component/file]"

## Rules
- Never rename, move, or restructure without a written proposal first
- Never change a public function signature without explicit approval
- Always propose before executing

## Workflow

### Step 1 — Understand current state
- Read the target file(s) in full
- Identify what the refactor should achieve (caller asked, or identified via code review)

### Step 2 — Write a proposal
Produce a proposal with:
- **Goal:** what problem is being solved
- **Scope:** which files will change
- **Changes:** bullet list of specific modifications
- **Risk:** what could break and how it is mitigated
- **Test plan:** how to verify the refactor is correct

### Step 3 — Wait for approval
Do not proceed until the user explicitly approves the proposal.

### Step 4 — Execute
Apply changes incrementally. Run tests after each logical group of changes.

### Step 5 — Verify
Confirm tests pass and invite the user to review the diff.

## Output Format

```
## Refactor Proposal: <title>

**Goal:** ...
**Scope:** <files affected>

**Changes:**
- ...

**Risk:** ...
**Test plan:** ...

Awaiting approval before proceeding.
```
