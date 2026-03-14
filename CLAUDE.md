# CLAUDE.md — Repository Memory File

> **READ THIS FIRST.** This is your operating guide for this repository.
> Do not modify any code, rename any files, or restructure any directories
> until you have completed the Repository Entry Protocol in
> `.claude/skills/entry-protocol.md`.

---

## 1. Project Identity

**Project Name:** `dqn-reconstruction`
**Purpose (WHY):** Reconstruct a Deep Q-Network trading agent from a surviving paper and report after the original implementation became unavailable — preserving confirmed hyperparameters exactly and tracking all unconfirmed decisions as declared assumptions.
**Status:** `Active Development`
**Primary Language(s):** `Python 3.11`
**Framework(s):** `PyTorch, Gymnasium`
**Owner / Portfolio:** `BUDDY26`

---

## 2. Repository Map (WHAT)

```
dqn-reconstruction/
├── .bootstrap-complete
├── .claude/
│   ├── hooks/
│   │   └── hooks.md
│   └── skills/
│       ├── code-review.md
│       ├── documentation.md
│       ├── entry-protocol.md
│       ├── qa-checklist.md
│       └── refactor-playbook.md
├── .env.example
├── .gitignore
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug-report.md
│   │   └── feature-request.md
│   ├── dependabot.yml
│   ├── pull_request_template.md
│   └── workflows/
│       └── ci.yml
├── CLAUDE.md
├── README.md
├── docs/
│   ├── architecture.md          # System design and component breakdown
│   ├── evidence-ledger.md       # Confirmed facts, gaps, and assumptions
│   ├── implementation-plan.md   # Planned modules and pipeline
│   ├── adr/
│   │   └── ADR-001-template.md  # Reconstruction philosophy ADR
│   ├── qa/
│   │   └── qa-plan.md
│   └── runbooks/
│       └── operations.md
├── scripts/
│   ├── bootstrap.sh
│   └── validate-structure.sh
├── src/                         # Implementation modules (added during coding phase)
│   └── .gitkeep
└── tests/
    ├── integration/
    │   └── .gitkeep
    └── unit/
        └── .gitkeep
```

<!-- Run `tree -L 3 --gitignore` and update above after src/ is populated -->

**Key Entry Points:**
- `src/train.py` — main training script (to be created)
- `src/agent.py` — DQN agent with epsilon-greedy policy (to be created)

**Configuration Files:**
- `.env.example` — environment variable reference (never commit `.env`)
- `src/config.py` — hyperparameter registry (to be created; all confirmed values live here)

**Test Suite:**
- `tests/` — pytest, run with `pytest tests/ -v`

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
- Deleting any file
- Creating new top-level directories
- Changing any confirmed hyperparameter value (see docs/evidence-ledger.md Category 1)

### 🚫 Never Do
- Commit or push to any branch
- Execute `rm -rf` or any irreversible destructive command
- Modify `.env` files or embed secrets in source code
- Run `DROP TABLE`, truncate databases, or execute destructive SQL
- Merge branches or create releases
- Silently substitute a confirmed hyperparameter with a different value

### Common Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run the training loop
python src/train.py

# Run tests
pytest tests/ -v

# Run linter + formatter
ruff check src/ && black --check src/

# Validate repository structure
bash scripts/validate-structure.sh
```

---

## 4. Architecture Summary

A Deep Q-Network agent trained to issue discrete buy/hold/sell decisions for five equity assets (AAPL, INTC, META, TQQQ, TSLA) using daily OHLCV data with engineered VWAP and percentage-change features normalized via StandardScaler. The Q-network is a two-layer feed-forward network (256→256 ReLU) backed by an experience replay buffer (capacity 100,000, batch size 64) and a separate target network for training stability. The agent is trained over 200 episodes using epsilon-greedy exploration decaying from 1.0 to 0.01, then evaluated on the held-out test period (2023-01-01 to 2024-01-01) using ROI, Sharpe Ratio, Maximum Drawdown, and Calmar Ratio.

> Full system design, component breakdown, and data flow are documented in
> `docs/architecture.md`. Key technical decisions are in `docs/adr/`.
> Confirmed vs. assumed implementation details are tracked in `docs/evidence-ledger.md`.

---

## 5. Known Issues / Sharp Edges

- **Evidence-ledger discipline** — Any implementation detail not in `docs/evidence-ledger.md`
  Category 1 (confirmed) must be added to Category 2 (gap) or Category 3 (assumption) before
  the dependent code is written. Do not skip this step.
- **Confirmed hyperparameters are constraints** — Do not adjust learning rate (0.0003),
  gamma (0.99), epsilon range (1.0→0.01), batch size (64), replay buffer (100,000), or
  training episodes (200) without an explicit ADR explaining the deviation.
- **Unresolved gaps block implementation** — `env.py` cannot be written until the reward
  function, observation window, and position sizing rules are resolved. See
  `docs/implementation-plan.md` for the full blocking dependency map.
- **Batch size note** — batch size of 64 is confirmed from the paper. This is on the smaller
  end for DQN; if training instability is observed, document it rather than silently changing
  the value.

---

## 6. Skills Available

| Skill | File | Purpose |
|-------|------|---------|
| Entry Protocol | `.claude/skills/entry-protocol.md` | **Run first** — mandatory scan before any changes |
| Code Review | `.claude/skills/code-review.md` | Structured review with severity-labeled findings |
| Refactor Playbook | `.claude/skills/refactor-playbook.md` | Safe, proposal-first refactoring workflow |
| Documentation | `.claude/skills/documentation.md` | Docstrings, README, architecture docs, ADRs |
| QA Checklist | `.claude/skills/qa-checklist.md` | Test coverage + portfolio readiness audit |

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
| Evidence Ledger | `docs/evidence-ledger.md` | Confirmed facts, gaps, and reconstruction assumptions |
| Implementation Plan | `docs/implementation-plan.md` | Planned modules, pipeline, and dependency order |
| ADR Index | `docs/adr/` | All architectural decision records |
| QA Plan | `docs/qa/qa-plan.md` | Test strategy and coverage map |
| Operations Runbook | `docs/runbooks/operations.md` | Setup, deployment, and troubleshooting |

---

## 9. Portfolio Context

**Target Audience:** Graduate admissions reviewers (UT Austin MSCS), software engineering employers
**Demonstrates:** `reinforcement learning, deep Q-network implementation, custom Gymnasium environment design, financial data preprocessing with feature engineering, evidence-bounded software reconstruction, documentation-first engineering, hyperparameter-faithful reproduction of academic experiments`
**Key Technical Decisions:** See `docs/adr/` for documented rationale
**Portfolio Repository:** Yes — maintain professional commit history and documentation standards

---

*Last updated by Claude: `2026-03-14`*
*Entry protocol completed: `yes — documentation phase complete, implementation phase pending`*
