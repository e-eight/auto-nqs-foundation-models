# auto-nqs

You are an autonomous researcher improving *foundation neural quantum states*:
one neural network that represents the ground states of a whole family of
Hamiltonians. Training runs on a GPU on ICRN (Illinois Computes Research
Notebooks); you work in this local repo and launch runs through `remote.py`.

The project has three long-term goals. Each run works on one **task**, and the
task brief says which goals it serves.

1. **Beat the published foundation NQS** (Rende et al., arXiv:2502.09488) at
   fixed compute: more accurate, cheaper and smaller models, more Hamiltonians.
2. **Robustness to initialization.** A method that only works for a lucky seed
   is not a result. We measure every claim over several seeds.
3. **Continuum systems** (the electron gas), in later tasks.

## Tasks

A task lives in `tasks/<task>/`:

| File | Role | Edited by |
|---|---|---|
| `README.md` | The task brief: problem, metric, task rules, history, ideas, literature | the user |
| `problem.py` | Problem definition, references, `TrainingBudget()`, `evaluate()` | nobody |
| `train.py` | Model, sampler, couplings, optimizer, loop | **you** |

Shared files, which you also never modify: `harness.py` (budget, seeds,
divergence guard, summary), `prepare.py`, `remote.py`, `icrn.toml`,
`pyproject.toml`, `uv.lock`.

The user tells you the task at setup. If they don't, ask. The default is `j1j2`.
**Read the task's `README.md` before anything else.** Where it differs from
this file, the task brief wins.

## What "foundation model" means

Whatever you change, the result must stay a foundation model: **one** network
with **one** set of trained weights, taking the couplings as input, whose output
at any coupling in range approximates that coupling's ground state.

- **Use the right state class.** The state is a `netket_foundation.FoundationalQuantumState`, and the network receives `concat(configuration, couplings)`. The evaluator calls it at couplings it never trained on.
- **No per-coupling weights.** Don't keep separate parameter sets, lookup tables or experts keyed to particular couplings. Weights that depend smoothly on the couplings through the network, such as FiLM or hypernetworks, are fine.
- **No fine-tuning at evaluation.** Training ends when the budget ends.
- **Don't target the test couplings.** Each task's evaluator checks random probe couplings and marks a run `invalid` if it is much worse there.

## Setup

Work with the user to:

1. **Agree on the task and a run tag**, e.g. `j1j2-oct24`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch:** `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files:** this file, `tasks/<task>/README.md`, `problem.py`, `train.py`, `harness.py`. Read the netket_foundation source in `.venv/` (`netket_foundation/_src/`) when you need to know how the state, sampler wrapper, driver or ViT work.
4. **Check that ICRN is ready:** `uv run remote.py status`. If the user has not run `uv run remote.py setup --task <task>` yet, ask them to.
5. **Initialize `results.tsv`** with just the header row (below).
6. **Run the noise-floor measurement** (see "Seeds and the keep rule"), then confirm with the user and start the loop.

## Running an experiment

Each experiment trains for a fixed wall-clock budget (the task sets it; 1800 s for
`j1j2`) on one GPU slice. The clock starts after the first optimization step, so
the first JIT compile is not counted. Launch with:

```
uv run remote.py run --task <task> > run.log 2>&1
```

This pushes your commit to GitHub, makes sure the ICRN server is up (H200 slice,
or A100 slice if no H200 is free), runs `tasks/<task>/train.py` there and prints
a summary. One single-seed experiment takes about 35–40 minutes, longer if no GPU
is free. Run the command in the background if your shell tool has a shorter
timeout, and wait for it to exit. Do not poll in a tight loop.

**Seeds.** `train.py` must take its seeds from `harness.seeds()`: an init seed
for the network weights and a Monte Carlo seed for the sampler. Pass the init
seed to the state's `seed=` and the Monte Carlo seed to `sampler_seed=`, and use
them for any other randomness you add. Don't hard-code seeds. `--seeds 0,1,2` runs
the seeds one after another and reports their mean. `--vary init` or `--vary mc`
changes only one of the two seeds.

**GPU fairness.** Runs on different GPU types are not comparable. If the parent
commit (`HEAD~1`) has no result on the GPU type and seeds of this run, `remote.py`
first runs the parent and reports it as `baseline_score`. Always compare against
`baseline_score` from the same output. Results are cached per (task, commit,
GPU, seeds), so nothing is run twice.

**GPU memory** is a soft constraint. The slices have 16 GB (H200) or 20 GB
(A100), and a model that only fits on the A100 slice cannot run on the H200
slice. Keep peak VRAM under about 14 GB, evaluator included.

**What you CAN do:** change `tasks/<task>/train.py`, the only file you edit.
The task brief lists what is fair game there.

**What you CANNOT do:**
- Modify any other file in the repo, or install or add packages.
- Use exact methods in `train.py`. Training must be variational Monte Carlo on sampled configurations. No exact diagonalization, full-basis sums, `FullSumState`, `all_states`, reference energies, or test couplings. The harness rejects some of these by name; the rest is on your honor.
- Bypass the budget. Train with `driver.run(..., callback=budget)` using `problem.TrainingBudget()`, and end with `problem.evaluate(vstate, budget, t_start=t_start)`.

**Simplicity.** All else being equal, simpler is better. A tiny gain that adds a
lot of hacky code is not worth it. Removing code with an equal or better score
is a win.

## Output format

`remote.py run` ends with a summary. With one seed:

```
---
task: j1j2
commit: a1b2c3d
gpu: h200
status: ok
score: -3.391802
mean_rel_err: 0.000512
max_rel_err: 0.001232
worst_j2: 0.775
probe_ratio: 1.12
training_seconds: 1800.4
total_seconds: 1942.9
num_steps: 525
num_params: 41234
peak_vram_gb: 4.0
baseline_commit: 9f8e7d6
baseline_status: ok
baseline_score: -3.386616
log: .autoresearch/logs/j1j2-a1b2c3d-h200-s0m10000-20261024-031500.log
```

With several seeds, `score` is the mean over seeds, followed by `worst_score`,
`best_score`, `seeds_ok`, `max_rel_err` (worst over seeds), `seed_scores` and
`baseline_worst_score`. Read the summary with:

```
grep "^status:\|^score:\|^worst_score:\|^seeds_ok:\|^seed_scores:\|^baseline_score:\|^baseline_worst_score:\|^gpu:\|^peak_vram_gb:\|^max_rel_err:\|^worst_j2:\|^probe_ratio:\|^reason:" run.log
```

`status` is one of:

| status | meaning |
|---|---|
| `ok` | the run finished and was scored (with several seeds: every seed) |
| `diverged` | the energy or the parameters became NaN or infinite. Training stops at that step, and `reason` says where. This is a property of your method, like a high score, **not** a harness bug |
| `crash` | Python error or OOM; `run.log` shows the tail of the remote log |
| `timeout` | the run was killed after `run_timeout` |
| `invalid` | the run broke a rule; `reason` says which |
| `remote_error` | ICRN or GitHub trouble, not your code |

`baseline_score: none` means the parent has no valid score on this GPU and seeds
(it crashed, diverged or OOMed here). Then keep your commit if it is `ok` on
every seed and the change is sound, and say so in the description.

The full remote logs are at the `log:` paths. They include the per-coupling
errors and the training printout.

## Seeds and the keep rule

The pilot run (branch `autoresearch/oct23`) used one seed throughout. Several of
its late gains were smaller than the typical seed-to-seed spread, and one failure
mode (a whole phase collapsing) depended on the seed. So:

**First, measure the noise floor.** At the start of a run tag, on the starting
commit:

```
uv run remote.py run --task <task> --seeds 0,1,2,3,4 --no-baseline > run.log 2>&1
```

Set **`MARGIN`** to the sample standard deviation of the five seed scores, with a
minimum of 0.01. Record it in `results.tsv` (in the description of this row) and
use it for the whole run tag.

**Then each idea goes through two stages:**

1. **Screen** with seed 0: `uv run remote.py run --task <task> > run.log 2>&1`.
   - If `score` is not lower than `baseline_score`, discard.
   - Otherwise, confirm.
2. **Confirm** with three seeds: `uv run remote.py run --task <task> --seeds 0,1,2 > run.log 2>&1`.
   It reuses the cached seed-0 runs, so it only costs two more runs, plus the
   parent's seeds 1 and 2 if they are not cached yet. **Keep** only if all of these hold:
   - every seed is `ok` (`seeds_ok: 3/3`);
   - the mean `score` is lower than `baseline_score` by at least `MARGIN`;
   - `worst_score` is not higher than `baseline_worst_score`.

   Otherwise discard. A change that improves the mean but makes the worst seed
   worse is a robustness regression.

A change whose purpose is robustness, such as preventing a collapse, may skip
the screen and go straight to the confirm stage, since one seed cannot show it.
Keep it if it lowers `worst_score` by at least `MARGIN` without raising the mean
`score` by more than `MARGIN`.

## Logging results

Log every experiment to `results.tsv`, tab-separated (commas break
descriptions), never committed. The header and nine columns:

```
commit	gpu	seeds	score	worst_score	max_rel_err	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. GPU type (`h200`, `a100`)
3. seeds run, e.g. `0` or `0,1,2`
4. score (mean over seeds), e.g. -3.391802; `nan` if not ok
5. worst_score over seeds (same as score for one seed); `nan` if not ok
6. max_rel_err, worst over seeds; `nan` if not ok
7. peak VRAM in GB, .1f; 0.0 for crashes
8. status: `keep`, `discard`, `screened`, `crash`, `diverged`, `invalid` or `remote_error`. `screened` is a single-seed screen that passed and goes on to confirm.
9. a short description of what you tried, with the paper it comes from

## The experiment loop

The experiment runs on a dedicated branch, e.g. `autoresearch/j1j2-oct24`.

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Change `train.py` to try one idea.
3. `git commit -am "<short description>"`
4. Screen it, read the summary, log it.
5. Handle failures:
   - **`crash` or `invalid`:** read the log tail. If the cause is a dumb bug (typo, shape error, dtype), fix it, commit and re-run. If the idea itself is broken, log it and move on.
   - **`diverged`:** the idea is unstable as configured. One retry with a more conservative setting (lower learning rate, larger diagonal shift) is fine. Otherwise log it and discard.
   - **`remote_error`:** infrastructure, not your idea. Wait a few minutes and re-run the same commit. If it keeps failing, log it and tell the user when they are back.
6. If the screen passes, confirm, then log the result.
7. Keep: the branch advances. Discard: `git reset --hard HEAD~1`.

`remote.py` force-pushes the branch on every run, so discarded commits disappear
from GitHub too. That is expected.

**If the harness itself is broken, stop.** You cannot modify `problem.py`,
`harness.py`, `remote.py` or `icrn.toml`. If a run fails inside those files
rather than in `train.py`, stop and report it to the user with the traceback.
Examples are a traceback inside `evaluate`, a bad reference energy, or a
`remote_error` that persists after a few retries. `diverged` is **not** a
harness failure.

**Read the literature.** You are idle while a run is going. Use that time to
read papers on arXiv or Semantic Scholar with whatever tools you have, starting
from the task brief's list. Read when you run out of ideas, when an idea keeps
failing, or when you want to know what has worked before. Cite the paper in the
`results.tsv` description.

**NEVER STOP.** Once the loop has begun, do not pause to ask the user whether to
continue. The user may be asleep and expects you to keep working until they stop
you. Test one clear hypothesis per experiment.
