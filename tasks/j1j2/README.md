# Task `j1j2`: J1–J2 Heisenberg, 4×6, one coupling

Goals served: **1** (accuracy and cost at fixed compute) and **2** (robustness
to initialization). Read `program.md` first. This brief adds the task-specific
details.

## The problem

Spin-1/2 J1–J2 Heisenberg model on a 4×6 periodic square lattice, total Sz = 0,
J1 = 1, with J2 as the model's input coupling over [0.2, 0.9]. The model is
scored at ten held-out couplings, J2 = 0.325, 0.375, …, 0.775. They sit between
the usual training couplings and cover the frustrated region near J2 ≈ 0.5.

Each score is computed **exactly**: the evaluator sums over all 2.7M Sz = 0
configurations and compares with Lanczos ground-state energies. There is no Monte
Carlo noise in a single run's score, and a correct wavefunction cannot go below
the exact energy. The variation between runs comes from training: the init and
Monte Carlo seeds.

**Metric.** `score` is the mean over the ten test couplings of log10(relative
energy error), so −3.0 means a typical relative error of 1e-3. It is on a log
scale, so improving the worst coupling (`worst_j2`, `max_rel_err`) matters as
much as polishing the easy ones.

**Budget.** 1800 s of training per run (600 s until 2026-10-24; at 600 s the model got only ~180 SR steps, so the search favoured fast convergence over accuracy).

## Task rules

- The final training couplings (`vstate.parameter_array`) must stay at least 0.01 away from every test coupling. Resampling couplings randomly during training is fine.
- Three random probe couplings are checked each run. A model more than 10× worse at a probe than at the neighbouring test couplings is `invalid`; `probe_ratio` shows how close you came.
- `TEST_J2`, `_ExactBasis`, `_exact_basis` and `reference_energies` are banned names in `train.py`.

## What you can change in `train.py`

Everything:
- **architecture:** ViT depth and width, patching, how J2 enters, custom Flax models;
- **output:** complex amplitudes, sign structure;
- **symmetries:** translations, point group, spin flip;
- **sampler:** chains, exchange moves, parallel tempering;
- **training couplings:** grid density, random resampling, curricula;
- **optimizer:** SR, NTK/MinSR, diagonal shift and learning-rate schedules, SPRING momentum;
- **initialization:** scales, zero-init heads, pre-training phases inside the budget;
- **precision and chunk sizes.**

## Starting point

`train.py` starts from the best model of the pilot (`autoresearch/oct23`, commit
`0f70b75`, score −3.39 on an H200 with one seed). Its sampler seed is now set
explicitly, so its seed-0 score will differ a little from the pilot's. The model:

- an own float32 ViT (float64 parameters), 2 layers, d=32, 8 heads, row tokens, factored attention;
- two complex heads, ψ = exp(f1)·Marshall + exp(f2), with a learnable J2 "tilt" between them;
- symmetrized over 4 x-translations × x-mirror × spin flip (16 images);
- 28 replicas on a 0.025 grid offset from the test couplings, 180 samples each;
- NTK-SR with diag_shift 1e-4, warmup + cosine learning rate peaking at 6e-2;
- Polyak weight averaging over the last 25 % of the budget.

## What the pilot learned (single seed, 600 s budget, so treat with care)

Worked, roughly in order of size: x-translation and spin-flip symmetrization; the
Marshall-sign head as a superposition (not a fixed sign); 28 replicas instead of
8 at the same total samples; float32 compute spent on more samples; weight
averaging; a higher peak learning rate.

Didn't work: a fixed Marshall sign; Marshall sign gated by a sigmoid in J2;
diag_shift 1e-5 or 2e-4; SPRING momentum 0.9; 2×2 plaquette tokens; J2 rescaled
to [−1, 1]; FiLM in every LayerNorm; 3 or 4 layers (too few steps in 600 s); 56
replicas; adding the y-mirror (32 images, too slow); fewer or more samples per
replica than 180.

**The known failure mode.** Three times, the stripe side (J2 ≳ 0.6) collapsed to
errors around 4e-2 while the rest stayed near 1e-3: after switching to float32,
after rewriting the ViT, and after going to 8 heads. The learnable tilt,
initialised at J2 = 0.55, prevented it for seed 0. Whether it prevents it for
other seeds is an open question, and the first thing the noise-floor run will
show.

## Research directions for this task

**Robustness (goal 2).** These usually need the confirm stage directly:
- the collapse rate over seeds (how often any coupling ends 10× above the median error), and whether the tilt really fixes it;
- `--vary init` against `--vary mc`: which seed matters;
- initialization schemes: smaller output-head init, zero-init residual branches, the tilt's initial center and slope, a learned versus fixed sign mixture;
- optimizer stability: diag_shift schedules (large early, small late), warmup length, gradient or update clipping;
- curricula in J2: start in the easy Néel region and widen, or start everywhere.

**Accuracy and cost (goal 1).**
- how J2 enters the network: a separate coupling token, per-layer conditioning, a hypernetwork for the output head;
- more steps per second: cheaper symmetrization (fewer images early, all late), mixed precision, sample counts;
- smaller models at equal score. Log `num_params` and note Pareto improvements in the description, even when the score is only equal.

## Literature

Prefer methods with evidence on frustrated 2D magnets. Starting points; follow
their references and citations:

- Rende, Viteritti, Becca, Scardicchio, Laio, Carleo, *Foundation neural-networks quantum states as a unified Ansatz for multiple Hamiltonians*, Nat. Commun. 16, 7213 (2025), arXiv:2502.09488. The method this repo implements; the target to beat.
- Viteritti, Rende, et al., *Quantum spin glass in the two-dimensional disordered Heisenberg model via foundation neural-network quantum states*, arXiv:2507.05073 (2025).
- Viteritti, Rende, Becca, *Transformer variational wave functions for frustrated quantum spin systems*, PRL 2023. The ViT ansatz on J1–J2.
- Rende, Viteritti, Bardone, Becca, Goldt, *A simple linear algebra identity to optimize large-scale neural network quantum states*, Commun. Phys. 2024. The NTK form of SR (`use_ntk=True`).
- Chen, Heyl, *Empowering deep neural quantum states through efficient optimization*, Nat. Phys. 2024. MinSR.
- Goldshlager, Abrahamsen, Lin, *A Kaczmarz-inspired approach to accelerate the optimization of neural network wavefunctions*, J. Comput. Phys. 2024. SPRING, the `momentum` option of `VMC_SR`.
- Choo, Neupert, Carleo, *Two-dimensional frustrated J1–J2 model studied with neural network quantum states*, PRB 2019. Symmetries and sign structure.
- Nomura, Imada, *Dirac-type nodal spin liquid revealed by refined quantum many-body solver using neural-network wave function, correlation ratio, and level spectroscopy*, PRX 2021. Symmetry projection and the J2 ≈ 0.5 region.
- Wu et al., *Variational benchmarks for quantum many-body problems*, Science 2024. V-score and how hard different regimes are.
- The Ketbook pages at https://ketworks.org (J1–J2 square lattice, transformer wavefunctions, optimization).
