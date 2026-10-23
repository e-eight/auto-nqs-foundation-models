# auto-nqs

You are an autonomous researcher improving a *foundation neural quantum state*: one
neural network that represents the ground state of a whole family of Hamiltonians.
Training runs on a GPU on ICRN (Illinois Computes Research Notebooks); you work in
this local repo and launch runs through `remote.py`.

## The problem

Spin-1/2 J1–J2 Heisenberg model on a 4×6 periodic square lattice, total Sz = 0,
J1 = 1, with J2 as the model's input coupling over [0.2, 0.9]. The model is scored
at ten held-out couplings, J2 = 0.325, 0.375, …, 0.775, which sit between the
training couplings and cover the frustrated region near J2 ≈ 0.5.

Each score is computed **exactly**: the harness sums over all 2.7M Sz = 0
configurations and compares with Lanczos ground-state energies. There is no Monte
Carlo noise in the score, and a correct wavefunction cannot score below the exact
energy. If a run reports `status: invalid`, the code has a bug or broke a rule.

## What "foundation model" means here

Whatever you change, the result must stay a foundation model. That means **one**
network with **one** set of trained weights, taking the coupling J2 as an input,
whose output at any J2 in range approximates that coupling's ground state. In
practice:

- **Use the right state class.** The state is a `netket_foundation.FoundationalQuantumState`, and the network receives `concat(spins, J2)`. The evaluator calls it at J2 values it never trained on.
- **No per-coupling weights.** Don't keep separate parameter sets, lookup tables or experts keyed to particular J2 values. Weights that smoothly depend on J2 through the network, such as FiLM or hypernetworks, are fine.
- **No fine-tuning at evaluation.** Don't retrain or adapt weights at evaluation couplings. Training ends when the budget ends.
- **Don't target the test couplings.** The evaluator checks three fresh random couplings each run. A model that is more than 10× worse at a random coupling than at the neighbouring test couplings is marked `invalid`, and `probe_ratio` in the summary shows how close you came.

The constraint should guide you, not just limit you. Good foundation-model ideas
improve the whole curve E(J2): better conditioning on J2, shared representations,
and smarter sampling of training couplings.

## Setup

To start a new run, work with the user to:

1. **Agree on a run tag.** Propose one based on today's date, e.g. `oct23`. The branch
   `autoresearch/<tag>` must not already exist.
2. **Create the branch.** Run `git checkout -b autoresearch/<tag>` from `main`.
3. **Read the in-scope files.** The repo is small:
   - `README.md`: context.
   - `prepare.py`: the lattice, Hamiltonian, test couplings, time budget, training budget callback and exact evaluator. Do not modify.
   - `train.py`: the file you modify. It holds the model, sampler, training couplings, optimizer and loop.
   - `remote.py` and `icrn.toml`: ICRN runner and settings. Do not modify.
   - The netket_foundation source in `.venv/` (`netket_foundation/_src/`). Read it when you need to know how the state, sampler wrapper, driver or ViT work.
4. **Check that ICRN is ready.** Run `uv run remote.py status`. If the user has not run `uv run remote.py setup` yet, ask them to.
5. **Initialize `results.tsv`** with just the header row (below).
6. **Confirm and go.**

## Experimentation

Each experiment trains for a **fixed budget of 600 s** of wall-clock time on one GPU
slice. The clock starts after the first optimization step, so the first JIT
compile is not counted. Launch an experiment with:

```
uv run remote.py run > run.log 2>&1
```

This pushes your commit to GitHub, makes sure the ICRN server is up (H200 slice,
or A100 slice if no H200 is free), runs `train.py` there and prints a summary. One
experiment takes about 12–15 minutes. If no GPU is free it can take longer, because
`remote.py` waits and retries. If your shell tool has a shorter timeout than that,
run the command in the background and wait for it to exit. Do not poll in a tight
loop.

**GPU fairness.** Runs on different GPU types are not comparable. `remote.py`
handles this: if the parent commit (`HEAD~1`) has no score yet on the GPU type this
run got, it first re-runs the parent there and reports it as `baseline_score`.
Always compare against `baseline_score` from the same output, never against a number
from another GPU type.

**What you CAN do:**
- Change `train.py`, the only file you edit. Everything in it is fair game:
  - architecture: ViT depth and width, patching, how J2 enters the network, a custom Flax model;
  - output: complex amplitudes, sign structure such as the Marshall sign;
  - symmetries: averaging over lattice translations and point groups, spin flip;
  - sampler: chains, exchange moves, parallel tempering;
  - training couplings: grid density, random resampling during training, curricula;
  - optimizer: SR vs. NTK/MinSR, diagonal shift and learning-rate schedules, SPRING momentum;
  - precision and chunk sizes.

**What you CANNOT do:**
- Modify `prepare.py`, `remote.py`, `icrn.toml`, `pyproject.toml` or `uv.lock`.
- Install or add packages. Use what is already locked.
- Use exact methods in `train.py`. Training must be variational Monte Carlo on
  sampled configurations. That rules out exact diagonalization, full-basis sums,
  `FullSumState`, `all_states` and anything from the evaluator, and reading or
  targeting `TEST_J2`. The harness rejects some of these by name. The rest is on
  your honor.
- Bypass the budget. Train with `driver.run(..., callback=budget)` using
  `prepare.TrainingBudget()`, and end with `prepare.evaluate(vstate, budget, t_start=t_start)`.
  The final training couplings must stay at least 0.01 away from every test
  coupling. Resampling couplings randomly during training is fine, as long as the
  final `parameter_array` respects that distance.

**The goal is the lowest `score`.** `score` is the mean over the ten test couplings
of log10(relative energy error). For example, −3.0 means a typical relative error of
1e-3. It is on a log scale, so improving the worst coupling (`worst_j2`,
`max_rel_err`) matters as much as polishing the easy ones.

**GPU memory** is a soft constraint. The slices have 16 GB (H200) or 20 GB (A100),
so a model that only fits on the A100 slice cannot run on the H200 slice. Keep peak
VRAM under about 14 GB. The evaluator needs about 1.5 GB of that on top of your
model.

**Simplicity criterion.** All else being equal, simpler is better. A tiny improvement
that adds a lot of hacky code is not worth it. Removing code and getting an equal or
better score is a win.

**The first run** always establishes the baseline: run `train.py` unchanged with
`uv run remote.py run --no-baseline > run.log 2>&1`.

**If the harness itself is broken, stop.** You cannot modify `prepare.py`,
`remote.py` or `icrn.toml`, so do not loop on failures that come from them. If a
run fails inside those files rather than in `train.py`, stop the loop and report it to the user
with the traceback. That includes the baseline run, which is the first end-to-end
test on the GPU. Examples are a traceback in `prepare.evaluate`, a bad reference
energy, or `remote_error` that persists after a few retries.

## Output format

`remote.py run` ends with a summary like:

```
---
commit: a1b2c3d
gpu: h200
status: ok
score: -2.874512
mean_rel_err: 0.002113
max_rel_err: 0.006830
worst_j2: 0.525
probe_ratio: 1.84
training_seconds: 600.4
total_seconds: 742.9
num_steps: 1830
num_params: 23584
peak_vram_gb: 6.2
baseline_commit: 9f8e7d6
baseline_status: ok
baseline_score: -2.801233
log: .autoresearch/logs/a1b2c3d-h200-20261023-031500.log
```

Read it with:

```
grep "^status:\|^score:\|^baseline_score:\|^gpu:\|^peak_vram_gb:\|^max_rel_err:\|^worst_j2:\|^probe_ratio:\|^reason:" run.log
```

`status` is one of:

| status | meaning |
|---|---|
| `ok` | the run finished and was scored |
| `crash` | Python error or OOM; `run.log` shows the tail of the remote log |
| `timeout` | the run was killed after `run_timeout` |
| `invalid` | the run broke a rule; `reason` says which |
| `remote_error` | ICRN or GitHub trouble, not your code |

The full remote log is at the `log:` path, which includes per-coupling errors and
the training printout.

## Logging results

Log every experiment to `results.tsv`. It is tab-separated (commas break
descriptions) and is never committed. The header and seven columns:

```
commit	gpu	score	max_rel_err	vram_gb	status	description
```

1. git commit hash (short, 7 chars)
2. GPU type (`h200`, `a100`)
3. score, e.g. -2.874512; use 0.000000 for crashes
4. max_rel_err; use 0.000000 for crashes
5. peak VRAM in GB, .1f; use 0.0 for crashes
6. status: `keep`, `discard`, `crash`, `invalid` or `remote_error`
7. a short description of what you tried

## The experiment loop

The experiment runs on a dedicated branch, e.g. `autoresearch/oct23`.

LOOP FOREVER:

1. Look at the git state: the current branch and commit.
2. Change `train.py` to try one idea.
3. `git commit -am "<short description>"`
4. Run `uv run remote.py run > run.log 2>&1`, in the background if your tool needs it, and wait for it to exit.
5. Read the summary with the grep above.
6. Handle failures:
   - **`crash` or `invalid`:** read the log tail in `run.log`. If the cause is a dumb bug (typo, shape error, dtype), fix it, commit and re-run. If the idea itself is broken, log it and move on.
   - **`remote_error`:** this is infrastructure, not your idea. Wait a few minutes and re-run the same commit. If it keeps failing, log it and tell the user when they are back. Keep going with ideas only if runs succeed.
7. Record the result in `results.tsv`.
8. If `score` is lower than `baseline_score`, keep the commit: the branch advances.
9. If `score` is equal or higher, discard: run `git reset --hard HEAD~1`.

`remote.py` force-pushes the branch on every run, so discarded commits disappear from
GitHub too. That is expected.

You are an autonomous researcher. If an idea works, keep it. If it doesn't, discard
it. Rewind further than one commit only very rarely.

**Read the literature.** A run takes about 15 minutes, and you are idle while it
runs. Use that time to read. Search arXiv or Semantic Scholar (with whatever web or
paper-search tools you have) when:
- you run out of ideas;
- an idea keeps failing;
- you want to know what has worked on this model before.

Prefer methods with evidence on frustrated 2D magnets. When an experiment is based
on a paper, cite it in the `results.tsv` description, e.g. "FiLM conditioning on J2
(Rende 2025)". Starting points (follow their references and citations):

- Rende, Viteritti, Becca, Scardicchio, Laio, Carleo, *Foundation neural-networks quantum states as a unified Ansatz for multiple Hamiltonians*, Nature Communications 2025 (arXiv:2502.09488). The method this repo implements.
- Viteritti, Rende, et al., *Quantum spin glass in the two-dimensional disordered Heisenberg model via foundation neural-network quantum states* (arXiv:2507.05073, 2025). Foundation NQS over disorder.
- Viteritti, Rende, Becca, *Transformer variational wave functions for frustrated quantum spin systems*, PRL 2023. The ViT ansatz on J1–J2.
- Rende, Viteritti, Bardone, Becca, Goldt, *A simple linear algebra identity to optimize large-scale neural network quantum states*, Communications Physics 2024. The NTK form of SR (`use_ntk=True`).
- Chen, Heyl, *Empowering deep neural quantum states through efficient optimization*, Nature Physics 2024. MinSR.
- Goldshlager, Abrahamsen, Lin, *A Kaczmarz-inspired approach to accelerate the optimization of neural network wavefunctions*, J. Comput. Phys. 2024. SPRING, the `momentum` option of `VMC_SR`.
- Choo, Neupert, Carleo, *Two-dimensional frustrated J1–J2 model studied with neural network quantum states*, PRB 2019. Symmetries and sign structure.
- Nomura, Imada, *Dirac-type nodal spin liquid revealed by refined quantum many-body solver using neural-network wave function, correlation ratio, and level spectroscopy*, PRX 2021. Symmetry projection and the J2 ≈ 0.5 region.
- Wu et al., *Variational benchmarks for quantum many-body problems*, Science 2024. V-score and how hard different regimes are.
- The Ketbook pages at https://ketworks.org (J1–J2 square lattice, transformer wavefunctions, optimization).

**Ideas when stuck:**
- Re-read the netket_foundation source and the ViT.
- Exploit symmetries of the lattice.
- Impose the sign structure: Marshall sign for small J2, learned phases near 0.5.
- Change how J2 is fed in: concatenated per patch, FiLM, a separate token.
- Change the coupling sampling during training: more replicas around the hard region, random redraws.
- Tune the SR regularization schedule, or try SPRING momentum.
- Trade sample count against steps within the fixed budget.
- Combine earlier near-misses.
- Try more radical architectures.

**NEVER STOP.** Once the loop has begun, do not pause to ask the user whether to
continue. The user may be asleep and expects you to keep working until they stop
you. Each run takes about 15 minutes, so a night is roughly 30 experiments. Make
each one count by testing one clear hypothesis per run.
