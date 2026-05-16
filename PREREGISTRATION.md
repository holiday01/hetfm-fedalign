# Pre-registration — heterogeneous-FM FL confirmatory analysis

**FROZEN: 2026-05-16T04:21:54Z.** This plan is fixed before any of its runs
execute. It is not edited after freeze. All listed outcomes are reported
regardless of direction or significance. No parameter tuning, no peeking
at partial results, no seed changes after this timestamp.

## Rationale
Earlier RQ1/RQ2/RQ3/RQ4 evidence (seeds 42–61) is **exploratory**: RQ2's
n=20 was run after seeing the n=5 trend (forking-paths), and the pre-
registered Wilcoxon test is unsatisfiable at n=5 (two-sided floor 0.0625).
This batch is a clean confirmation on **fresh, never-executed seeds** with
a sample size at which Wilcoxon can reject.

## Frozen configuration (headline, locked 2026-05-16)
Linear projector, k=256, tie=fm (per-FM-type-tied), λ_proto=1, λ_con=1,
150 rounds. LB (local-only mixed-federation floor) = **0.1283** (Week-1,
fixed). Scheme for RQ1/RQ2 = `balanced`. A-control = same method,
`homogeneous:Conch_v15`, same config (FM-homogeneous ⇒ scheme-independent).
FedProtoKD-B = tuned variant (aggregate_projector=True), `balanced`.

## Seeds
**62–91 inclusive (n=30).** Verified never previously executed (prior runs
used 42–61 only). Each seed runs: method-linear (balanced), A-linear
(homog CONCH), FedProtoKD-B (balanced).

## Hypotheses, tests, decision rules
Primary test = **Wilcoxon signed-rank, two-sided, paired across the 30
seeds, α = 0.05** (valid: n=30 floor ≪ 0.05). Secondary (reported
alongside, not gating): paired-t p, mean difference with 95% bootstrap CI
(10k resamples), per-seed sign count.

- **H1 (RQ1 recovery).** mean over 30 seeds of closure
  = (method−LB)/(A−LB) ≥ 0.90. Pre-registered bar = 0.90. Report mean,
  std, per-seed; "saturated" flag if >1 (means method ≥ control: report
  as "recovers/matches", not "closes N%").
- **H2 (RQ2 complementarity).** method-linear > A-linear. Reject H0 if
  Wilcoxon p < 0.05 AND mean Δ > 0. Non-uniformity disclosed (sign count);
  H2 "uniform" only if additionally all 30 seeds positive (not required
  for the headline; reported either way).
- **H3 (incrementality vs FedProtoKD).** method-linear > FedProtoKD-B
  (balanced). Reject H0 if Wilcoxon p < 0.05 AND mean Δ > 0.

## Reporting commitment
The exploratory seeds 42–61 results stay reported AS exploratory. This
n=30 batch is the confirmatory headline. Every H1/H2/H3 outcome is
reported with its real numbers whatever they are, including if any
weakens or fails. Output: `hetfm/results/prereg_confirm.json`. Driver:
`hetfm/run_prereg.py` (executes exactly this plan, no other knobs).
