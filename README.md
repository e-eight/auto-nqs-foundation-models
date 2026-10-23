# auto-nqs-foundation-models

[autoresearch](https://github.com/karpathy/autoresearch) adapted to
[netket_foundation](https://github.com/netket/netket_foundation). An AI agent edits
`train.py` to improve a foundation neural quantum state for the J1–J2 Heisenberg
model. Each experiment trains for a fixed 10 minutes on an ICRN GPU slice. The agent
keeps changes that lower the exact held-out energy error and discards the rest.

## Files

| File | Role | Edited by |
|---|---|---|
| `prepare.py` | Lattice (4×6 PBC, Sz = 0), parametrized Hamiltonian, test couplings, `TrainingBudget`, exact evaluator, Lanczos references | nobody |
| `train.py` | Model, sampler, training couplings, optimizer, loop | the agent |
| `program.md` | Instructions for the agent | you |
| `remote.py` | Pushes the commit to GitHub and runs it on ICRN through the JupyterHub API | nobody |
| `icrn.toml` | Hub URL, user, server name, GPU resource options, timeouts | you |

## Metric

`score` = mean over J2 ∈ {0.325, 0.375, …, 0.775} of log10((E − E0)/|E0|). Lower is
better.

- **E** is the model's variational energy at that coupling. It is computed by exact
  summation over all 2,704,156 Sz = 0 configurations, so the score has no sampling
  noise and can't be gamed below the true ground state.
- **E0** is the Lanczos ground-state energy, computed once by `prepare.py` and cached
  in `~/.cache/auto-nqs/`.

The test couplings are held out: they sit between the baseline's training couplings,
so the score measures what the foundation model learned about the whole family, not
how well it fits a few points.

## How it runs

```
local (agent + git)                         ICRN (JupyterHub, default server)
───────────────────                         ───────────────────────────────
git commit train.py
uv run remote.py run ── git push --force ─► github.com/e-eight/auto-nqs-foundation-models
         │                                              │
         ├── Hub API: start server (H200 → A100) ──► ~/auto-nqs/default: git fetch + checkout <sha>
         ├── kernel websocket: launch detached ───►  uv sync --frozen --extra cuda; uv run train.py
         ├── poll every 30 s ◄────────────────────   ~/auto-nqs/runs/<sha>-<gpu>-<time>/run.log
         └── prints summary; caches score per (commit, GPU type) in .autoresearch/
```

- **Code** moves only through git.
- **Logs** come back over the kernel connection.
- **The token** is read from `~/.config/icrn/token` and is never printed.

## One-time setup

1. **GPU resource values.** These are set in `icrn.toml` (`3_h200`, `2_a100`), read
   from running sessions.
2. **GitHub on ICRN.** `~/.pixi/bin/gh` must be logged in (`gh auth status`) with
   access to this repo. `uv` is used from `PATH`, `~/.local/bin` or `~/.pixi/bin`, and
   installed with `pixi global install uv` if it is missing.
3. **Push `main`, then run setup:**
   ```
   git push -u origin main
   uv run remote.py setup
   ```
   This starts the GPU server, clones the repo, installs the locked environment
   with CUDA JAX, and computes the reference energies.
4. **Start the agent** in this directory (e.g. `claude`), with permissions set so it
   can run `git` and `uv run remote.py` unattended, and prompt:
   > Have a look at program.md and let's kick off a new experiment! Let's do the setup first.

Other commands: `uv run remote.py status`, `uv run remote.py stop`. To run two
agents on two GPUs, give the second one `--server nqs` (a named server) and a
separate branch.

## Notes

- **GPU slices.** ICRN GPUs are MIG slices: H200 with 16 GB, A100 with 20 GB. Each
  is slower than a whole GPU, and the two are not comparable, so `remote.py` re-scores
  the parent commit when a run lands on a different GPU type.
- **Pinned versions.** `netket` and `netket-foundation` are pinned to exact git
  commits in `pyproject.toml`, because netket_foundation needs unreleased NetKet
  changes. `uv.lock` freezes the rest.
- **Changing the problem.** To change the lattice or couplings, edit the constants
  at the top of `prepare.py` on `main`. That starts a new, incomparable series of
  runs. Full enumeration works up to about 28 sites.
