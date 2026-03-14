# ADR-001: Evidence-Bounded Reconstruction as the Development Philosophy

**Date:** 2026-03-14
**Status:** Accepted
**Deciders:** Ruben Aleman (BUDDY26)

---

## Context

This repository reconstructs a Deep Q-Network trading agent whose original implementation became
unavailable. The surviving artifacts are a final project report and the original research paper.
Together, these sources confirm a subset of the technical details required to reproduce the
experiment: the asset universe, date range, input features, preprocessing method, network
architecture, and a set of training hyperparameters.

A significant number of implementation details are not specified in either source. These include
the reward function formulation, observation window size, epsilon decay schedule, optimizer choice,
target network update frequency, position sizing rules, and transaction cost assumptions.

Without a governing philosophy, reconstruction efforts risk silently filling these gaps with
conventional DQN defaults, producing a system that passes superficial scrutiny but does not
faithfully represent the original experiment. This matters for the stated purpose of the
repository: producing a reproducible and academically honest reconstruction of a specific prior
experiment, not a generic DQN implementation.

Three approaches were considered:

1. **Unrestricted reconstruction** — implement a working DQN trading agent using best-practice
   defaults wherever the paper is silent, without tracking which details were sourced vs. assumed.
2. **Evidence-bounded reconstruction** — treat confirmed details as constraints, explicitly
   track all gaps, and require every unconfirmed choice to be declared as an assumption before
   it enters the code.
3. **Deferred reconstruction** — halt all implementation until every gap is resolved from the
   original source, accepting that some gaps may never be filled.

---

## Decision

We will implement **evidence-bounded reconstruction** as the governing philosophy of this project.

This means:

1. **Confirmed evidence is authoritative.** Values taken directly from the paper or report are
   treated as constraints. They are not tuned, adjusted, or substituted without a new ADR
   explaining the deviation.

2. **Gaps are tracked, not hidden.** Every implementation detail not sourced from the paper or
   report is first recorded in `docs/evidence-ledger.md` as a confirmed gap before any code
   is written that depends on it.

3. **Assumptions are declared, not inferred.** When a gap must be filled to proceed, the chosen
   value is added to `docs/evidence-ledger.md` Category 3 with a justification and risk
   assessment. Assumptions are never silently embedded in code.

4. **Documentation precedes implementation.** No module is written until its design dependencies
   are resolved in the evidence ledger. The module dependency order is: `config.py` →
   `data.py` → `env.py` → `network.py` → `replay_buffer.py` → `agent.py` → `train.py` →
   `evaluate.py`. No module is begun until the preceding module has passing unit tests.

5. **Deviations from confirmed values are flagged.** If a confirmed value produces unexpected
   behavior during implementation, the behavior is documented — not silently corrected.

---

## Rationale

### Why not unrestricted reconstruction?

An unrestricted reconstruction would produce a functional DQN agent, but would not be a
reconstruction of the original experiment. Untracked gaps become invisible substitutions. The
resulting repository would be indistinguishable from a generic implementation, defeating the
purpose of the reconstruction effort. For a portfolio reviewed by graduate admissions committees,
intellectual honesty about what is known versus assumed is more valuable than a polished system
with hidden decisions.

### Why not deferred reconstruction?

Some gaps — particularly the reward function and observation window — may not be resolvable from
the surviving documentation. Requiring full source fidelity before any implementation would
produce no working code. The evidence-bounded approach permits forward progress by making
assumptions explicit rather than blocking on them indefinitely.

### Why evidence-bounded reconstruction?

It preserves the academic integrity of the reconstruction while permitting a working
implementation. Every reader of the repository can distinguish confirmed behavior from assumed
behavior by consulting `docs/evidence-ledger.md`. This is the appropriate posture for a
reconstruction project where the original is unavailable.

### Alternatives Considered

| Alternative | Reason Rejected |
|-------------|----------------|
| Unrestricted reconstruction | Gaps become invisible; academic integrity compromised |
| Deferred reconstruction | Would block indefinitely on unfillable gaps |
| Hybrid (unrestricted for "minor" details) | The line between major and minor is subjective; creates inconsistency |

---

## Consequences

### Positive
- Every implementation choice is traceable to a source or an explicit assumption.
- The repository is honest about what is reconstructed faithfully and what is interpolated.
- Future readers (or a future session) can update assumptions if new evidence becomes available
  without searching through code comments.
- Reviewing the evidence ledger provides a clear map of reconstruction confidence.

### Negative / Trade-offs
- Implementation is slower — every gap must be addressed before the dependent module is written.
- The evidence ledger requires active maintenance; it is not self-updating.
- Some confirmed values (notably batch size = 64, which is atypically small for DQN) may produce
  training instability. Under this philosophy, those values are preserved and the behavior
  documented rather than substituted.

### Neutral
- This philosophy does not prescribe specific technical choices (optimizer, reward, etc.). Those
  remain to be resolved per gap.
- The evidence ledger is a living document that grows as gaps are resolved.

---

## References

- `docs/evidence-ledger.md` — implementation of this philosophy
- `docs/implementation-plan.md` — module dependency order
- Original research paper (internal — not publicly available)
- Final project report (internal — not publicly available)

---

*ADRs are immutable once Accepted. To supersede this record, create ADR-002 with status
"Supersedes ADR-001" and update this file's status to "Superseded by ADR-002".*
