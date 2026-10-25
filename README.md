# auto-nqs-foundation-models

[autoresearch](https://github.com/karpathy/autoresearch) adapted to
[netket_foundation](https://github.com/netket/netket_foundation). An AI agent edits
a task's `train.py` to improve a foundation neural quantum state. Each experiment
trains for a fixed time on an ICRN GPU slice. The agent screens each idea with one
seed, confirms it with three, and keeps only changes that improve the mean score
by more than the seed-to-seed noise without making the worst seed worse.

## Goals and roadmap

1. Improve on Rende et al., arXiv:2502.09488: accuracy, cost and model size at fixed compute, and more Hamiltonians.
2. Robustness to initialization: every claim is measured over seeds.
3. Continuum systems from [Ketboard](https://ketworks.org/#benchmarks).

| Phase | Task | Status |
|---|---|---|
| 0 | Harness: NaN → `diverged`, per-seed runs, multi-task layout | done |
| A | `j1j2`: J1–J2 4×6, J2 ∈ [0.2, 0.9]; noise floor, then accuracy and robustness | ready |
| B | `j1j2j3`: J1–J2–J3 on 4×6, a 2D grid of test couplings, exact scoring | planned |
| B | size transfer: train on several lattice sizes, test on held-out sizes | planned |
| C | `heg`: 3D electron gas, 7+7 electrons, simple cubic, Γ point, rs as the coupling | needs a feasibility check |

Phase C needs harness work before an agent can use it. netket_foundation's
`ParametrizedOperator` only handles discrete spaces, NetKet has no Ewald
summation, and the references (TC-FCIQMC, DMC) are not exact. So the score is a
Monte Carlo estimate with error bars, and it can go below the reference.

## Files

| File | Role | Edited by |
|---|---|---|
| `program.md` | Instructions for the agent, common to all tasks | you |
| `tasks/<task>/README.md` | Task brief: problem, rules, history, ideas, literature | you |
| `tasks/<task>/problem.py` | Problem definition, references, `TrainingBudget()`, evaluator | nobody |
| `tasks/<task>/train.py` | Model, sampler, training couplings, optimizer, loop | the agent |
| `harness.py` | Seeds, NaN-aware training budget, rule checks, summary block | nobody |
| `prepare.py` | `uv run prepare.py --task <task>`: compute and cache references | nobody |
| `remote.py` | Pushes the commit to GitHub and runs it on ICRN (JupyterHub API) or DeltaAI (Slurm over SSH) | nobody |
| `icrn.toml` | Hub URL, user, server name, GPU resource options, timeouts | you |
| `deltaai.toml` | SSH master socket, remote paths, Slurm account and resources, seeds per GPU | you |

## Metric (task `j1j2`)

`score` = mean over J2 ∈ {0.325, 0.375, …, 0.775} of log10((E − E0)/|E0|). Lower is
better. With several seeds, `score` is the mean over seeds and `worst_score` the
highest.

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
git commit tasks/<task>/train.py
uv run remote.py run ── git push --force ─► github.com/e-eight/auto-nqs-foundation-models
         │                                              │
         ├── Hub API: start server (H200 → A100) ──► ~/auto-nqs/default: git fetch + checkout <sha>
         ├── kernel websocket: launch detached ───►  uv sync; NQS_SEED=… uv run tasks/<task>/train.py
         ├── poll every 30 s ◄────────────────────   ~/auto-nqs/runs/<task>-<sha>-<gpu>-<seeds>-<time>/run.log
         └── prints summary; caches results per (task, commit, GPU type, seeds) in .autoresearch/
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
   uv run remote.py setup --task j1j2
   ```
   This starts the GPU server, clones the repo, installs the locked environment
   with CUDA JAX, and computes the reference energies.
4. **Start the agent** in this directory (e.g. `claude`), with permissions set so it
   can run `git` and `uv run remote.py` unattended, and prompt:
   > Have a look at program.md and let's kick off a new experiment on task j1j2! Let's do the setup first.

Other commands: `uv run remote.py status`, `uv run remote.py stop`.

**Two agents at once** (for example on two tasks): give each one its own working
directory, branch and ICRN server.
1. Commit everything on `main` first, including `.claude/settings.json`: a new worktree only gets committed files.
2. Create the second checkout: `git worktree add ../nqs-agent2 main`.
3. Start the second agent in `../nqs-agent2`, and tell it to add `--server nqs` (the named server) to every `remote.py` command.

Each checkout then has its own `.autoresearch/` score cache, `run.log` and
`results.tsv`. Two agents in one directory would reset and switch each other's
branches. Run `remote.py setup` for a task once, before starting the agents:
the reference-energy cache on ICRN is shared.

**DeltaAI.** `--backend deltaai` (or `NQS_BACKEND=deltaai`) runs the same
commands as Slurm jobs on NCSA DeltaAI, with the settings in `deltaai.toml`:
1. Open an SSH master connection, since every new DeltaAI login asks for Duo:
   `ssh -fN -o ControlMaster=yes -o ControlPersist=yes -o ControlPath=~/.ssh/cm-deltaai soham@dtai-login.delta.ncsa.illinois.edu`.
2. `uv run remote.py check --backend deltaai` tests the connection, shows the
   balance and validates the job settings with `sbatch --test-only`. It runs nothing.
3. `uv run remote.py setup --backend deltaai --task j1j2`, then `run` as usual.

Each job is one GH200 that runs `seeds_per_gpu` seeds of one commit side by side.
All jobs of a `run`, baseline included, are queued at once, and the summary ends
with the GPU-hours billed. Packing changes the steps a run gets, so it is part of
the GPU tag (`gh200`, `gh200x4`, `gh200x4mps`), and `--seeds` must be a multiple
of `seeds_per_gpu`. An interrupted `run` leaves its jobs running; the same
command re-attaches to them.

To run a task locally (CPU JAX), set `PYTHONPATH` so that `harness` and `tasks`
import: `PYTHONPATH=. uv run tasks/j1j2/train.py`.

## Notes

- **GPU slices.** ICRN GPUs are MIG slices: H200 with 16 GB, A100 with 20 GB. Each
  is slower than a whole GPU, and the two are not comparable, so `remote.py` re-scores
  the parent commit when a run lands on a different GPU type.
- **Pinned versions.** `netket` and `netket-foundation` are pinned to exact git
  commits in `pyproject.toml`, because netket_foundation needs unreleased NetKet
  changes. `uv.lock` freezes the rest.
- **Changing or adding a problem.** Add a new task directory instead of editing an
  existing `problem.py`: a changed problem starts a new, incomparable series of
  runs. Full enumeration works up to about 28 sites.
- **NaNs.** Training stops at the first non-finite sampled energy, and the run
  reports `status: diverged` instead of a NaN score. `remote.py` never uses a
  run that isn't `ok` as a baseline.
