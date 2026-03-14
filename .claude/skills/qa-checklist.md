# QA Checklist

**Purpose:** Test coverage audit and portfolio readiness check.

## When to Run
Say: "Run QA" or "Portfolio readiness check"

## Test Coverage Checklist

- [ ] Unit tests exist for all public functions in `src/`
- [ ] Integration tests cover end-to-end flows
- [ ] Edge cases are tested (empty inputs, boundary values, error paths)
- [ ] Tests are deterministic (no random seeds without fixture control)
- [ ] `pytest tests/ -v` passes with 0 failures
- [ ] Coverage is reported (run with `--cov=src`)

## Code Quality Checklist

- [ ] No `TODO` or `FIXME` comments in production code paths
- [ ] No unused imports
- [ ] All public functions have docstrings
- [ ] Linter passes (`ruff check src/`)
- [ ] Formatter passes (`black --check src/`)

## Documentation Checklist

- [ ] `README.md` reflects current functionality
- [ ] `docs/architecture.md` is up to date
- [ ] At least one ADR exists in `docs/adr/`
- [ ] `CLAUDE.md` sections 1–9 are fully filled in (no `{{PLACEHOLDER}}` tokens)

## Portfolio Readiness Checklist

- [ ] Repository has a descriptive, professional README
- [ ] CI pipeline passes on `main`
- [ ] Commit history is clean and messages are meaningful
- [ ] No secrets, credentials, or `.env` files committed
- [ ] `scripts/validate-structure.sh --strict` passes

## Output Format

```
## QA Report

### Test Coverage
<findings>

### Code Quality
<findings>

### Documentation
<findings>

### Portfolio Readiness
<findings>

**Overall:** Ready / Not Ready
**Blockers:** <list or "none">
```
