# autoresearch/oct23 records

Results of the autoresearch run on 2026-10-23: 50 experiments improving a
foundation neural quantum state for the J1–J2 Heisenberg model on a 4×6
periodic lattice (see `program.md` on `main`). All runs used an H200 slice.

- `results.tsv`: one row per experiment, in the order they ran: commit, GPU,
  score (mean log10 relative energy error at the ten test couplings, lower is
  better), max relative error, peak VRAM, status, and the change tried.
- `logs/`: the full remote log of every run (including the baseline re-runs),
  named `<commit>-<gpu>-<time>.log`, with per-coupling energies, probe
  results and the training timing breakdown.
- `scores.json`: the per-(commit, GPU) score cache written by `remote.py`.

Every experiment's code is tagged `oct23/expNN` (01–50), including the
discarded ones; the tag message repeats the row from `results.tsv`. The best
run is experiment 46, commit `0f70b75`, score −3.392, which is the head of
`autoresearch/oct23`.
