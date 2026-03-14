# Operations Runbook

> **Status:** Stub — fill in as the project matures.

---

## Prerequisites

<!-- List what must be installed before setup. -->

- Python 3.x
- pip

---

## Setup

```bash
# Clone
git clone <repo-url>
cd dqn-reconstruction

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Project

```bash
# Fill in after implementation
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Linting and Formatting

```bash
# Fill in after tooling is configured
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError`
**Cause:** Virtual environment not activated
**Fix:** Run `source .venv/bin/activate`

---

## CI/CD

The CI pipeline runs on every push and pull request to `main`.
See `.github/workflows/ci.yml` for the full pipeline definition.

---

*Last updated: <!-- date -->*
