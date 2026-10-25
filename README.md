# autoresearch/j1j2-oct24 records

Results of the autoresearch run of 2026-10-24/25: 13 experiments improving a
foundation neural quantum state for the J1–J2 Heisenberg model on a 4×6
periodic lattice (see `program.md` and `tasks/j1j2/` on `main`). All runs used
an H200 slice and the 1800 s training budget (about 535 SR steps per run).

This run used the multi-seed protocol: a 5-seed noise floor on the starting
commit (`4876d26`, the `988e940` model from j1j2-oct23), a seed-0 screen for
each idea, and a confirm stage on seeds 0–2. The noise floor was mean −3.559,
worst seed −3.523, with no collapse on any seed; its seed stdev set
MARGIN = 0.037. Confirm runs were compared with the noise floor's seeds 0–2
(mean −3.551, worst −3.536), and worst-seed score was tracked as a robustness
metric.

- `results.tsv`: one row per screen or confirm run, in the order they ran:
  commit, GPU, seeds, score (mean log10 relative energy error at the ten test
  couplings, lower is better; for several seeds, the mean over seeds), worst
  seed score, max relative error, peak VRAM, status, and the change tried.
- `logs/`: the full remote log of every run, named
  `j1j2-<commit>-<gpu>-s<init seed>m<MC seed>-<time>.log`, with per-coupling
  energies, probe results and the training timing breakdown.
- `scores.json`: the per-(task, commit, GPU, seeds) score cache written by
  `remote.py`, for this task and run only.

Every experiment's code is tagged `j1j2-oct24/expNN` (01–13), including the
discarded ones; the tag message repeats the final `results.tsv` row for that
commit. The best kept run is experiment 09, commit `f73c5c0`
(`n_discard_per_chain` 5 → 0): mean −3.581, worst seed −3.574 over seeds 0–2.
Its mean gain (0.030) is under MARGIN, but it was kept because the worst seed
improved by 0.038 and every seed improved.

`f73c5c0` is the head of `autoresearch/j1j2-oct24`. Later ideas (more samples,
DECAY_STEPS 2250, DECAY_STEPS 2250 + y-mirror) did not clear MARGIN on top of
it and survive only in their tags.

Findings worth carrying forward:

- DECAY_STEPS 2250 and the y-mirror symmetry each gave a small, consistent
  gain (~0.02) but neither cleared MARGIN, and stacked they did not add up.
- Three layers can reach the best single-seed score (−3.656) but collapse on
  the stripe side on some seeds; the stripe sign prior and zero-init residuals
  did not make depth robust.
- SAMPLES_PER_REPLICA 360 does not fit the 16 GB slice; 270 fits but runs 1.7×
  slower and ends worse.
