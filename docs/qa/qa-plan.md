# QA Plan

> **Status:** Stub — fill in after implementation begins.

---

## Test Strategy

### Unit Tests
**Location:** `tests/unit/`
**Framework:** pytest
**Scope:** Individual functions and classes in `src/`
**Coverage target:** ≥ 80% line coverage

### Integration Tests
**Location:** `tests/integration/`
**Framework:** pytest
**Scope:** End-to-end flows (e.g., training loop, environment interaction)

---

## Test Coverage Map

| Module | Unit Tests | Integration Tests | Coverage |
|--------|-----------|-------------------|----------|
| `src/` | — | — | — |

---

## Running Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Strict (fail on warnings)
pytest tests/ -v -W error
```

---

## Known Gaps

<!-- List any areas intentionally not tested and why. -->

---

*Last updated: <!-- date -->*
