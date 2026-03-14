# Hooks

**Purpose:** Automatic guardrails that apply during every Claude Code session.

These hooks represent behavioral rules, not executable scripts. Claude checks them before taking the associated actions.

---

## Active Hooks

### 1. `post-edit-format`
**Trigger:** After editing any `.py`, `.ts`, or `.js` file
**Action:** Remind the user to run the formatter before committing
**Message:** "Remember to run `ruff check src/ && black src/` (or equivalent) before committing."

---

### 2. `pre-delete-guard`
**Trigger:** Before deleting any file
**Action:** Halt and require explicit confirmation
**Message:** "Deletion requires explicit approval. Confirm you want to delete `<file>` and explain why."

---

### 3. `test-on-core-change`
**Trigger:** After editing any file under `src/`
**Action:** Remind to run the test suite
**Message:** "Core source was modified. Run `pytest tests/ -v` to verify nothing is broken."

---

### 4. `block-sensitive-dirs`
**Trigger:** Before modifying files in `auth/`, `billing/`, `infra/`, or `migrations/`
**Action:** Halt and require explicit approval
**Message:** "This directory is marked sensitive in CLAUDE.md. Explicit approval required before proceeding."

---

### 5. `no-secrets-in-code`
**Trigger:** Before writing any string literal that resembles an API key, token, password, or secret
**Action:** Replace with an environment variable reference pattern
**Message:** "Potential secret detected. Use an environment variable instead: `os.environ['VAR_NAME']`"

---

### 6. `proposal-before-refactor`
**Trigger:** Before renaming files, moving files, or changing function signatures
**Action:** Write a refactor proposal (see `.claude/skills/refactor-playbook.md`) and wait for approval
**Message:** "This action requires a written proposal first. Running refactor-playbook.md protocol."
