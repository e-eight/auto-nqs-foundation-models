# autoresearch/j1j2-oct23 records

Results of the autoresearch run of 2026-10-23/24: 21 experiments improving a
foundation neural quantum state for the J1–J2 Heisenberg model on a 4×6
periodic lattice (see `program.md` and `tasks/j1j2/` on `main`). All runs used
an H200 slice and the 600 s training budget (about 180 SR steps per run).

This run used the multi-seed protocol: a 5-seed noise floor on the starting
commit (`1422ee5`), a seed-0 screen for each idea, and a confirm stage on
seeds 0–4. MARGIN = 0.01. Seed 3 of the starting commit collapsed on the
stripe side, so worst-seed score was tracked as a robustness metric.

- `results.tsv`: one row per screen or confirm run, in the order they ran:
  commit, GPU, seeds, score (mean log10 relative energy error at the ten test
  couplings, lower is better; for several seeds, the mean over seeds), worst
  seed score, max relative error, peak VRAM, status, and the change tried.
- `logs/`: the full remote log of every run, named
  `j1j2-<commit>-<gpu>-s<init seed>m<MC seed>-<time>.log`, with per-coupling
  energies, probe results and the training timing breakdown.
- `scores.json`: the per-(task, commit, GPU, seeds) score cache written by
  `remote.py`, for this task only.

Every experiment's code is tagged `j1j2-oct23/expNN` (01–21), including the
discarded ones; the tag message repeats the final `results.tsv` row for that
commit. The best kept run is experiment 03, commit `988e940` (diag_shift
schedule 1e-2 → 1e-4): mean −3.306, worst seed −3.222 over seeds 0–4.

The branch `autoresearch/j1j2-oct23` was left at `80d0468` (experiment 21),
which was discarded; `988e940` is its parent.

After this run the training budget was raised to 1800 s, because many ideas
were limited by step count. Scores here are not comparable with later runs.
