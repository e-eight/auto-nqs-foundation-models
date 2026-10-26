# Task `j1j2_6x6`: J1–J2 Heisenberg, 6×6, one coupling

Goals served: **1** (accuracy and cost at fixed compute) and **2** (robustness
to initialization). Read `program.md` first. This brief adds the task-specific
details.

## The problem

Spin-1/2 J1–J2 Heisenberg model on a 6×6 periodic square lattice, total Sz = 0,
J1 = 1, with J2 as the model's input coupling over [0.2, 0.9]. The model is
scored at seven held-out couplings, J2 = 0.3, 0.4, …, 0.9: the couplings in
that range with published exact energies (Varbench). They are not the 4×6
task's test couplings, so 4×6 and 6×6 scores are not comparable. 6×6 is the
standard benchmark cluster for this model: it has the full square symmetry,
and the published results (Varbench, RBM+PP, Rende et al.'s foundation ViT at
J2 = 0.5) are given on it.

**Monte Carlo evaluation.** The Sz = 0 space has 9.1·10⁹ states, so the energy
cannot be summed exactly. The evaluator samples |ψ|² with its own sampler, not
the one in `train.py`: 1024 chains, 300 burn-in sweeps, then 2¹⁷ samples per
coupling. The sampler uses exchange moves plus whole-lattice symmetry moves
(translation, rotation, reflection, global spin flip). The energy therefore has
a statistical error: `score_err` reports it, typically ~0.005. A run is
`invalid` if the energy is more than 5 standard errors below the exact energy,
or if the evaluation chains did not mix (R̂ > 1.05).

**References.** Varbench's 6×6 exact diagonalization energies (Lanczos in the
fully symmetric sector), divided by 4 for spin-1/2 operators and hard-coded in
`problem.py`. At J2 = 0.5, E0/N = −0.5038097, as in Schulz, Ziman, Poilblanc
(1996).

**Metric.** `score` is the mean over the seven test couplings of log10(relative
energy error), so −3.0 means a typical relative error of 1e-3. It is on a log
scale, so improving the worst coupling (`worst_j2`, `max_rel_err`) matters as
much as polishing the easy ones. `mean_v_score` is the mean V-score
(N·Var E / E², Wu et al. 2024) over the test couplings, for comparison with
the literature.

**Order parameters (diagnostic, not scored).** At each test coupling the
evaluator also prints the Néel and stripe order parameters,
m²_Néel = C(π,π)/N and m²_stripe = [C(π,0) + C(0,π)]/(2N) with
C(k) = Σ_r e^{ik·r} ⟨S₀·S_r⟩, from the full S·S correlations on 2¹⁴ of the
evaluation samples (error ~0.001). They are compared with exact
diagonalization values from Rende et al. (Fig. 4a), and the summary reports
the largest deviation (`max_m2_err`, `worst_m2_j2`). Use them to tell *how* a
run fails, not as an objective: a model stuck in a Néel-like state on the
stripe side shows m²_Néel too large and m²_stripe too small at J2 ≥ 0.7,
while a model that is in the right phase but imprecise shows small
deviations everywhere. Exact values:

| J2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|---|
| m²_Néel | 0.1668 | 0.1384 | 0.0975 | 0.0462 | 0.0130 | 0.0064 | 0.0033 |
| m²_stripe | 0.0143 | 0.0148 | 0.0168 | 0.0403 | 0.1057 | 0.1201 | 0.1259 |

Rende et al.'s foundation model misses most near the transition: m²_Néel
0.102 at J2 = 0.5, m²_stripe 0.051 at J2 = 0.6 (read from the same figure).

**Budget.** 1800 s of training per run on one GH200 (DeltaAI). Evaluation
takes a few more minutes and is not counted.

## Task rules

- The final training couplings (`vstate.parameter_array`) must stay at least 0.01 away from every test coupling. Resampling couplings randomly during training is fine.
- Three random probe couplings are checked each run. There is no exact energy there, so they are compared by V-score: a model whose V-score at a probe is more than 10× worse than at the neighbouring test couplings is `invalid`. `probe_ratio` shows how close you came.
- `TEST_J2`, `REFERENCE_E0` and `REFERENCE_M2` are banned names in `train.py`.

## What you can change in `train.py`

Everything:
- **architecture:** ViT depth and width, patching, how J2 enters, custom Flax models;
- **output:** complex amplitudes, sign structure;
- **symmetries:** translations, point group (the 6×6 torus has all of D4), spin flip;
- **sampler:** chains, exchange moves, parallel tempering;
- **training couplings:** grid density, random resampling, curricula;
- **optimizer:** SR, NTK/MinSR, diagonal shift and learning-rate schedules, SPRING momentum;
- **initialization:** scales, zero-init heads, pre-training phases inside the budget;
- **precision and chunk sizes.**

## Starting point

`train.py` is the kept head of run tag `j1j2-oct24` on 4×6 (commit `f73c5c0`),
unchanged except that it imports this task's `problem.py`. On 4×6 it scored
mean −3.581, worst seed −3.574 over seeds 0–2 (ICRN H200 slice, exact
evaluation). The model:

- an own float32 ViT (float64 parameters), 2 layers, d=32, 8 heads, row tokens (6 sites each), factored attention;
- two complex heads, ψ = exp(f1)·Marshall + exp(f2), with a learnable J2 "tilt" between them;
- symmetrized over 6 x-translations × x-mirror × spin flip (24 images);
- 28 replicas on a 0.025 grid (0.2125, …, 0.8875), at least 0.0125 from every test coupling, 180 samples each, chains carried over between steps (`n_discard_per_chain = 0`);
- NTK-SR with diag_shift 1e-2 decaying exponentially to 1e-4 over 60 steps, warmup + cosine learning rate peaking at 6e-2 (`DECAY_STEPS = 750`);
- Polyak weight averaging over the last 25 % of the budget.

**Measured on one GH200 (no evaluation):** 1.00 s per step, so about 1790
steps in 1800 s; 56 % GPU use, 4.6 GB peak memory, 25 890 parameters. With
3 layers: 1.38 s per step (~1300 steps), 6.3 GB. With 360 samples per
replica: 2.05 s per step (~880 steps), 13.9 GB. Doubling the samples doubles
the sampling and local-energy time and makes the SR solve ~3.3× slower.

So the step-based schedules no longer match the step count: the learning rate
reaches its floor at step 750, well before the end of the budget, as it did
on 4×6. This head has never been scored on 6×6.

## What the 4×6 runs learned

The full records are on `autoresearch/j1j2-oct23-records` and
`autoresearch/j1j2-oct24-records` (results, logs), with every experiment's
code tagged `j1j2-oct23/expNN` and `j1j2-oct24/expNN`. In short:

- **Steps were the limit.** Raising the budget from 600 s to 1800 s improved the score from −3.31 to −3.56, more than any idea did. On 6×6 a run gets about 1790 steps on a GH200, against ~540 on the 4×6 ICRN slice, so ideas that lost on step count on 4×6 deserve a new test here: D_MODEL 48, 3–4 layers, the y-mirror (48 images on 6×6), more samples per replica, 2×2 plaquette tokens, 56 replicas, FiLM conditioning.
- **Not about step count, and failed on 4×6:** zero-init residual branches, a 10× smaller output-head init (all seeds collapsed), float32 parameters (SR 3× slower, then diverged), peak learning rate 8e-2 (bimodal over seeds), a fixed or sigmoid-gated Marshall sign, decaying the learning rate fully.
- **The failure mode.** The stripe side (J2 ≳ 0.6) can collapse to errors around 4e-2 while the rest stays near 1e-3. A large diag_shift early (1e-2 decaying to 1e-4 over 60 steps) removed it on 4×6; adding a J2 embedding to every token brought it back.
- **Seeds.** The init seed matters more than the Monte Carlo seed. Most single-seed gains of ~0.02 did not survive more seeds, while gains measured seed by seed (same seed, parent vs child) were consistent.
- **Hardest coupling.** On 4×6 the worst coupling was J2 = 0.575. Here the nearest test couplings are 0.5 and 0.6.

## Research directions for this task

**Accuracy and cost (goal 1).**
- Use the steps: depth, width, samples per replica, the y-mirror or full D4 symmetrization, and schedules (`DECAY_STEPS`, warmup, diag_shift) matched to ~1800 steps.
- How J2 enters the network: a separate coupling token, per-layer conditioning, a hypernetwork for the output head.
- More steps per second: cheaper symmetrization (fewer images early, all late), mixed precision, sample counts, chunk sizes now that memory is not a limit (96 GB).
- Smaller models at equal score. Log `num_params` and note Pareto improvements in the description, even when the score is only equal.

**Robustness (goal 2).**
- The collapse rate over seeds on the stripe side, and whether the diag_shift schedule still prevents it on 6×6. The order parameters show whether a bad coupling is in the wrong phase.
- Initialization schemes: the tilt's initial center and slope, a learned versus fixed sign mixture.
- Optimizer stability: diag_shift schedules, warmup length, gradient or update clipping.
- Curricula in J2: start in the easy Néel region and widen, or start everywhere.

## Reference results on 6×6

Relative errors at J2 = 0.5 (exact E/N = −0.503810, spin-1/2 units):

| Method | E/N | rel. error | Source |
|---|---|---|---|
| RBM+PP, K=0 + A1 + spin-parity projections | −0.503800 | 1.9e-5 | Nomura & Imada, PRX 2021 (Varbench) |
| RBM+PP, simplified point-group projection | −0.503765 | 8.9e-5 | Nomura & Imada, arXiv:2005.14142, App. B.1 |
| RBM, 72 hidden units, K=0 + A1 + spin-parity projections | −0.503760 | 9.9e-5 | Nomura, JPCM 2021 (Varbench) |
| DMRG, bond dimension 2048 | −0.502598 | 2.4e-3 | Varbench |
| RBM α=1, no projections | −0.486928 | 3.4e-2 | Varbench |

Rende et al. (arXiv:2502.09488) report ~1e-3 to 4e-3 relative error at
J2 = 0.5 for L ≥ 6 with a foundation ViT trained on 1000 couplings (8 layers,
d=72, 16 000 samples, 3500 steps): far more compute than one 1800 s run.

## Literature

Prefer methods with evidence on frustrated 2D magnets. Starting points; follow
their references and citations:

- Rende, Viteritti, Becca, Scardicchio, Laio, Carleo, *Foundation neural-networks quantum states as a unified Ansatz for multiple Hamiltonians*, Nat. Commun. 16, 7213 (2025), arXiv:2502.09488. The method this repo implements; the target to beat.
- Viteritti, Rende, Becca, *Transformer variational wave functions for frustrated quantum spin systems*, PRL 2023. The ViT ansatz on J1–J2.
- Rende, Viteritti, Bardone, Becca, Goldt, *A simple linear algebra identity to optimize large-scale neural network quantum states*, Commun. Phys. 2024. The NTK form of SR (`use_ntk=True`).
- Chen, Heyl, *Empowering deep neural quantum states through efficient optimization*, Nat. Phys. 2024. MinSR.
- Goldshlager, Abrahamsen, Lin, *A Kaczmarz-inspired approach to accelerate the optimization of neural network wavefunctions*, J. Comput. Phys. 2024. SPRING, the `momentum` option of `VMC_SR`.
- Choo, Neupert, Carleo, *Two-dimensional frustrated J1–J2 model studied with neural network quantum states*, PRB 2019. Symmetries and sign structure on 6×6 and 10×10.
- Nomura, Imada, *Dirac-type nodal spin liquid revealed by refined quantum many-body solver using neural-network wave function, correlation ratio, and level spectroscopy*, PRX 2021. Symmetry projection and the J2 ≈ 0.5 region; the best 6×6 energies.
- Schulz, Ziman, Poilblanc, *Magnetic order and disorder in the frustrated quantum Heisenberg antiferromagnet in two dimensions*, J. Phys. I 6, 675 (1996). The original 6×6 exact diagonalization.
- Wu et al., *Variational benchmarks for quantum many-body problems*, Science 2024. V-score and how hard different regimes are.
- The Ketbook pages at https://ketworks.org (J1–J2 square lattice, transformer wavefunctions, optimization).
