# CLAUDE.md — Repository Memory File

> **READ THIS FIRST.** This is your operating guide for this repository.
> Do not modify any code, rename any files, or restructure any directories
> until you have completed the Repository Entry Protocol in
> `.claude/skills/entry-protocol.md`.

---

## 1. Project Identity

**Project Name:** `{{PROJECT_NAME}}`
**Purpose (WHY):** `{{PROJECT_DESCRIPTION}}`
**Status:** `{{STATUS}}`  <!-- Active Development | Maintenance | Portfolio | Archived -->
**Primary Language(s):** `{{LANGUAGE}}`
**Framework(s):** `{{FRAMEWORK}}`
**Owner / Portfolio:** `{{GITHUB_USERNAME}}`

---

## 2. Repository Map (WHAT)

```
{{REPO_TREE}}
```

<!-- Run `tree -L 3 --gitignore` and paste the output above after first scan -->

**Key Entry Points:**
- `{{ENTRY_POINT_1}}`
- `{{ENTRY_POINT_2}}`

**Configuration Files:**
- `.env.example` — environment variable reference (never commit `.env`)
- `{{CONFIG_FILE_2}}`

**Test Suite:**
- `tests/` — {{TEST_FRAMEWORK}}, run with `{{TEST_COMMAND}}`

---

## 3. Rules + Commands (HOW)

### ✅ Allowed Without Asking
- Read any file
- Improve documentation (docstrings, comments, README, CLAUDE.md)
- Fix formatting and style inconsistencies
- Add or improve inline comments
- Add new test files in `tests/`
- Update `.env.example` with new variable names (never values)

### ⚠️ Requires Explicit Approval Before Executing
- Renaming or moving any file or directory
- Changing function signatures or public APIs
- Adding, removing, or upgrading dependencies
- Modifying database schemas or migration files
- Editing files in `src/auth/`, `src/billing/`, or `infra/`
- Deleting any file
- Creating new top-level directories

### 🚫 Never Do
- Commit or push to any branch
- Execute `rm -rf` or any irreversible destructive command
- Modify `.env` files or embed secrets in source code
- Run `DROP TABLE`, truncate databases, or execute destructive SQL
- Merge branches or create releases

### Common Commands
```bash
# Install dependencies
{{INSTALL_COMMAND}}

# Run the application
{{RUN_COMMAND}}

# Run tests
{{TEST_COMMAND}}

# Run linter + formatter
{{LINT_COMMAND}}

# Validate repository structure
bash scripts/validate-structure.sh
```

---

## 4. Architecture Summary

`{{ARCHITECTURE_SUMMARY}}`

> Full system design, component breakdown, and data flow are documented in
> `docs/architecture.md`. Key technical decisions are in `docs/adr/`.

---

## 5. Known Issues / Sharp Edges

<!-- Fill this in after running the entry protocol. Examples below: -->
- `{{SHARP_EDGE_1}}` — e.g., `src/auth/` token logic is tightly coupled to legacy session model. Do not refactor without approval.
- `{{SHARP_EDGE_2}}` — e.g., `migrations/` must be applied in strict order. Never reorder.

---

## 6. Skills Available

| Skill | File | Purpose |
|-------|------|---------|
| Entry Protocol | `.claude/skills/entry-protocol.md` | **Run first** — mandatory scan before any changes |
| Code Review | `.claude/skills/code-review.md` | Structured review with severity-labeled findings |
| Refactor Playbook | `.claude/skills/refactor-playbook.md` | Safe, proposal-first refactoring workflow |
| Documentation | `.claude/skills/documentation.md` | Docstrings, README, architecture docs, ADRs |
| QA Checklist | `.claude/skills/qa-checklist.md` | Test coverage + portfolio readiness audit |
| Release Procedure | `.claude/skills/release-procedure.md` | Steps before tagging a version |

---

## 7. Hooks Active

| Hook | Trigger | Action |
|------|---------|--------|
| `post-edit-format` | After editing `.py` / `.ts` / `.js` files | Suggest running formatter |
| `pre-delete-guard` | Before any file deletion | Halt and require explicit confirmation |
| `test-on-core-change` | After editing files in `src/` | Remind to run test suite |
| `block-sensitive-dirs` | Before modifying `auth/`, `billing/`, `infra/`, `migrations/` | Halt and require approval |
| `no-secrets-in-code` | Before writing string literals resembling keys/tokens | Replace with env variable pattern |
| `proposal-before-refactor` | Before renaming, moving, or changing signatures | Write proposal first |

---

## 8. Documentation Index

| Document | Location | Description |
|----------|----------|-------------|
| Architecture Overview | `docs/architecture.md` | Full system design and component breakdown |
| ADR Index | `docs/adr/` | All architectural decision records |
| QA Plan | `docs/qa/qa-plan.md` | Test strategy and coverage map |
| Operations Runbook | `docs/runbooks/operations.md` | Setup, deployment, and troubleshooting |
| API Reference | `docs/api-reference.md` | Endpoint documentation (create when applicable) |

---

## 9. Portfolio Context

**Target Audience:** Graduate admissions reviewers (UT Austin MSCS), software engineering employers
**Demonstrates:** `{{DEMONSTRATES}}`  <!-- e.g., REST API design, async processing, TDD -->
**Key Technical Decisions:** See `docs/adr/` for documented rationale
**Portfolio Repository:** Yes — maintain professional commit history and documentation standards

---

*Last updated by Claude: `{{LAST_UPDATED}}`*
*Entry protocol completed: `no — run on first session`*
