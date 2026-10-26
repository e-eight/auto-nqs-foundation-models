"""
Task j1j2_6x6: fixed problem definition, reference energies and Monte Carlo evaluator.

DO NOT MODIFY. The agent edits tasks/j1j2_6x6/train.py only.

Problem: spin-1/2 J1-J2 Heisenberg model on a 6 x 6 square lattice with
periodic boundaries, in the total Sz = 0 sector,

    H(J2) = J1 * sum_<ij> S_i . S_j + J2 * sum_<<ij>> S_i . S_j,   J1 = 1,

with spin-1/2 operators (not Pauli matrices). Sites are numbered row-major,
i = y * LX + x, so a row of LX sites is a contiguous block of the input.

A foundation model is trained on J2 in [J2_MIN, J2_MAX] and scored at the
held-out couplings TEST_J2. The Sz = 0 basis has 9.1e9 states, so the energy
is a Monte Carlo estimate with the evaluator's own sampler (EVAL_SAMPLES
samples per coupling), and the reference energies are exact diagonalization
results hard-coded below (REFERENCE_E0). Random probe couplings are checked
through the energy variance, which needs no reference.
"""

import functools
import math
import time

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf

import harness

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LX, LY = 6, 6  # columns, rows (periodic in both directions)
N_SITES = LX * LY
J1 = 1.0
J2_MIN, J2_MAX = 0.2, 0.9  # ParameterSpace range the foundation model covers
TEST_J2 = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # the couplings with published exact energies
MIN_TEST_DISTANCE = 0.01  # final training couplings must stay this far from TEST_J2
N_PROBES = 3  # fresh random couplings per evaluation, to catch overfitting to TEST_J2
PROBE_FACTOR = 10.0  # a probe's V-score may be at most this much worse than its neighbouring test points
TIME_BUDGET = 1800  # seconds of training, counted from the end of the first step
EVAL_SAMPLES = 2**17  # Monte Carlo samples per coupling
EVAL_CHAINS = 2**10
EVAL_DISCARD = 300  # sweeps discarded per chain (one sweep = N_SITES proposals)
EVAL_SYMMETRY_MOVES = 0.1  # fraction of proposals that apply a lattice symmetry and spin flip
MAX_R_HAT = 1.05  # chains this far from agreeing make the energy estimate untrustworthy
EVAL_BATCH = 256  # samples per local-energy batch
EVAL_CHUNK = 4096  # configurations per forward pass, when train.py sets no chunk_size
BELOW_SIGMAS = 5.0  # an energy this many standard errors below E0 is a bug

# Exact ground-state energies at TEST_J2: Varbench (github.com/varbench/varbench,
# J1J2/square_36_P_<J2>.md), from Lanczos with lattice-symmetries in the sector
# with momentum (0, 0), all reflections even and spin inversion even. Varbench
# uses Pauli matrices, so its energies are divided by 4 for spin-1/2 operators.
# E0/N at J2 = 0.5 is -0.5038097, as in Schulz, Ziman, Poilblanc (1996).
_VARBENCH_E0_PAULI = {
    0.3: -80.994163118435,
    0.4: -76.283280827996,
    0.5: -72.548590160286,
    0.6: -71.026357467734,
    0.7: -76.320176597454,
    0.8: -84.454070394730,
    0.9: -93.463489624054,
}
REFERENCE_E0 = {j2: e / 4 for j2, e in _VARBENCH_E0_PAULI.items()}
assert set(REFERENCE_E0) == set(TEST_J2)

# Exact order parameters (m2_neel, m2_stripe) at TEST_J2, a diagnostic only (not
# scored): the exact diagonalization curves of Rende et al., arXiv:2502.09488v3,
# Fig. 4a, read from the vertices of the vector figure in the arXiv source
# (images/structure_factor_J3=0.pdf), so about +-1e-4. Definitions (their Eq. 5):
# C(k) = sum_r e^{ik.r} <S_0 . S_r>, m2_neel = C(pi, pi) / N,
# m2_stripe = [C(pi, 0) + C(0, pi)] / (2 N).
REFERENCE_M2 = {
    0.3: (0.1668, 0.0143),
    0.4: (0.1384, 0.0148),
    0.5: (0.0975, 0.0168),
    0.6: (0.0462, 0.0403),
    0.7: (0.0130, 0.1057),
    0.8: (0.0064, 0.1201),
    0.9: (0.0033, 0.1259),
}
assert set(REFERENCE_M2) == set(TEST_J2)
CORR_SAMPLES = 2**14  # of each coupling's evaluation samples, used for the order parameters
CORR_BATCH = 64  # samples per batch (each needs N_SITES (N_SITES - 1) / 2 amplitudes)

# train.py may not use exact or reference machinery: training must be VMC.
BANNED_IN_TRAIN = harness.banned("TEST_J2", "REFERENCE_E0", "_VARBENCH_E0_PAULI", "REFERENCE_M2")

# ---------------------------------------------------------------------------
# Lattice, Hilbert space, Hamiltonian
# ---------------------------------------------------------------------------


def _site(x, y):
    return (y % LY) * LX + (x % LX)


def _bonds(offsets):
    bonds = set()
    for y in range(LY):
        for x in range(LX):
            for dx, dy in offsets:
                i, j = _site(x, y), _site(x + dx, y + dy)
                assert i != j
                bonds.add((min(i, j), max(i, j)))
    return tuple(sorted(bonds))


assert LX >= 3 and LY >= 3, "smaller periodic lattices double-count bonds"
NN_BONDS = _bonds([(1, 0), (0, 1)])
NNN_BONDS = _bonds([(1, 1), (1, -1)])


def graph():
    """Nearest-neighbour graph (edge color 0) plus next-nearest (color 1)."""
    edges = [(i, j, 0) for i, j in NN_BONDS] + [(i, j, 1) for i, j in NNN_BONDS]
    return nk.graph.Graph(edges=edges)


def hilbert():
    """Spin-1/2 lattice of N_SITES sites in the total Sz = 0 sector (values +-1)."""
    return nk.hilbert.Spin(0.5, N_SITES, total_sz=0)


def parameter_space():
    """One coupling, J2, in [J2_MIN, J2_MAX]."""
    return nkf.ParameterSpace(N=1, min=J2_MIN, max=J2_MAX)


_BOND_I = np.array([b[0] for b in NN_BONDS + NNN_BONDS])
_BOND_J = np.array([b[1] for b in NN_BONDS + NNN_BONDS])
_IS_NNN = np.array([0.0] * len(NN_BONDS) + [1.0] * len(NNN_BONDS))
_FLIP = np.ones((len(_BOND_I), N_SITES))
_FLIP[np.arange(len(_BOND_I)), _BOND_I] = -1
_FLIP[np.arange(len(_BOND_I)), _BOND_J] = -1


class _J1J2:
    """J1-J2 Hamiltonian at one (possibly traced) J2, exposing get_conn_padded."""

    def __init__(self, j2):
        self.j2 = j2

    def get_conn_padded(self, x):
        J = J1 * (1 - _IS_NNN) + self.j2 * _IS_NNN
        si, sj = x[..., _BOND_I], x[..., _BOND_J]
        diag = 0.25 * jnp.sum(J * si * sj, axis=-1)
        anti = si != sj
        # Exchanging an antiparallel pair == flipping both spins. Parallel pairs
        # keep x itself with a zero matrix element, so no out-of-sector input.
        x_swapped = jnp.where(anti[..., None], x[..., None, :] * _FLIP, x[..., None, :])
        mels_off = jnp.where(anti, 0.5 * J, 0.0)
        xs = jnp.concatenate([x[..., None, :], x_swapped.astype(x.dtype)], axis=-2)
        mels = jnp.concatenate([diag[..., None], mels_off], axis=-1)
        return xs, mels


def hamiltonian(hi, ps):
    """The parametrized J1-J2 Hamiltonian over the joint (spins, J2) space."""
    return nkf.operator.ParametrizedOperator(hi, ps, lambda p: _J1J2(p[0]))


# ---------------------------------------------------------------------------
# Training budget
# ---------------------------------------------------------------------------


def TrainingBudget():
    """The harness budget callback with this task's TIME_BUDGET. It also stops
    training as soon as the sampled energy is NaN or infinite."""
    return harness.TrainingBudget(TIME_BUDGET)


# ---------------------------------------------------------------------------
# Monte Carlo evaluation
# ---------------------------------------------------------------------------


def _with_coupling(s, j2):
    s = s.astype(float)
    return jnp.concatenate([s, jnp.full((*s.shape[:-1], 1), j2, dtype=s.dtype)], axis=-1)


def _lattice_symmetries():
    """Site permutations of the torus: translations times the point group D4."""
    assert LX == LY, "the point group of a non-square torus is smaller"
    mats = [(1, 0, 0, 1), (0, -1, 1, 0), (-1, 0, 0, -1), (0, 1, -1, 0)]  # rotations (a, b; c, d)
    mats += [(a, b, -c, -d) for a, b, c, d in mats]  # times the mirror y -> -y
    perms = []
    for a, b, c, d in mats:
        for tx in range(LX):
            for ty in range(LY):
                perms.append([_site(a * x + b * y + tx, c * x + d * y + ty) for y in range(LY) for x in range(LX)])
    return np.array(perms)


class _SymmetryMove(nk.sampler.rules.MetropolisRule):
    """Proposes a random lattice symmetry of the whole configuration, times a
    random global spin flip. Symmetric (the group is closed under inverses), so
    the chains can jump between symmetry-related modes of |psi|^2, which single
    exchanges cross slowly."""

    perms: jax.Array

    def __init__(self, perms):
        self.perms = jnp.asarray(perms)

    def transition(rule, sampler, machine, parameters, state, key, σ):
        kg, kf = jax.random.split(key)
        g = jax.random.randint(kg, σ.shape[:1], 0, rule.perms.shape[0])
        flip = jnp.where(jax.random.bernoulli(kf, 0.5, σ.shape[:1]), -1, 1).astype(σ.dtype)
        return jnp.take_along_axis(σ, rule.perms[g], axis=-1) * flip[:, None], None


def _eval_sampler():
    rules = (nk.sampler.rules.ExchangeRule(graph=graph(), d_max=1), _SymmetryMove(_lattice_symmetries()))
    rule = nk.sampler.rules.MultipleRules(rules, [1 - EVAL_SYMMETRY_MOVES, EVAL_SYMMETRY_MOVES])
    return nk.sampler.MetropolisSampler(hilbert(), rule, n_chains=EVAL_CHAINS)


@functools.cache
def _at_coupling(apply_fun):
    """apply_fun on spins alone, with the coupling passed in the variables, so the
    sampler compiles once for all couplings."""

    def f(variables, s):
        variables = dict(variables)
        j2 = variables.pop("eval_coupling")["j2"]
        return apply_fun(variables, _with_coupling(s, j2))

    return f


def _mc_energy(vstate, j2, seed):
    """Energy statistics of `vstate` at coupling j2, sampled from |psi|^2 by the
    evaluator's own Metropolis exchange sampler (not the one train.py set up),
    and the samples, (chains, samples per chain, N_SITES)."""
    apply_fun, variables = vstate._apply_fun, vstate.variables
    chunk = vstate.chunk_size or EVAL_CHUNK
    state = nk.vqs.MCState(
        _eval_sampler(),
        apply_fun=_at_coupling(apply_fun),
        variables={**variables, "eval_coupling": {"j2": jnp.asarray(j2, dtype=float)}},
        n_samples=EVAL_SAMPLES,
        n_discard_per_chain=EVAL_DISCARD,
        sampler_seed=seed,
    )
    samples = state.sample()
    eloc = _local_energies(apply_fun, variables, samples.reshape(-1, N_SITES), j2, chunk)
    return nk.stats.statistics(eloc.reshape(samples.shape[:-1])), samples


def _logpsi(apply_fun, variables, j2, chunk):
    return nk.jax.apply_chunked(lambda c: apply_fun(variables, _with_coupling(c, j2)), chunk_size=chunk)


@functools.partial(jax.jit, static_argnums=(0, 4))
def _local_energies(apply_fun, variables, x, j2, chunk):
    logpsi = _logpsi(apply_fun, variables, j2, chunk)

    def batch(xb):
        xs, mels = _J1J2(j2).get_conn_padded(xb)
        lp = logpsi(xb)
        lpc = logpsi(xs.reshape(-1, N_SITES)).reshape(xs.shape[:-1])
        return jnp.sum(mels * jnp.exp(lpc - lp[:, None]), axis=-1)

    return nk.jax.apply_chunked(batch, chunk_size=EVAL_BATCH)(x)


# All site pairs i < j, and the sign e^{ik.(r_i - r_j)} of each pair at
# k = (pi, pi), (pi, 0), (0, pi).
_PAIR_I, _PAIR_J = np.triu_indices(N_SITES, k=1)
_XY = np.array([(i % LX, i // LX) for i in range(N_SITES)])
_DXY = _XY[_PAIR_I] - _XY[_PAIR_J]
_PAIR_SIGN = np.stack([(-1.0) ** (_DXY[:, 0] + _DXY[:, 1]), (-1.0) ** _DXY[:, 0], (-1.0) ** _DXY[:, 1]])
_PAIR_FLIP = np.ones((len(_PAIR_I), N_SITES))
_PAIR_FLIP[np.arange(len(_PAIR_I)), _PAIR_I] = -1
_PAIR_FLIP[np.arange(len(_PAIR_I)), _PAIR_J] = -1


@functools.partial(jax.jit, static_argnums=(0, 4))
def _local_structure_factors(apply_fun, variables, x, j2, chunk):
    """Local estimators of C(k) / N at k = (pi, pi), (pi, 0), (0, pi), with
    C(k) / N = (1 / N^2) sum_{i, j} e^{ik.(r_i - r_j)} S_i . S_j, from the full
    S_i . S_j = S^z_i S^z_j + (S^+_i S^-_j + S^-_i S^+_j) / 2."""
    logpsi = _logpsi(apply_fun, variables, j2, chunk)

    def batch(xb):
        si, sj = xb[:, _PAIR_I], xb[:, _PAIR_J]
        anti = si != sj
        swapped = jnp.where(anti[..., None], xb[:, None, :] * _PAIR_FLIP, xb[:, None, :])
        lp = logpsi(xb)
        lpc = logpsi(swapped.reshape(-1, N_SITES)).reshape(swapped.shape[:-1])
        sisj = 0.25 * si * sj + jnp.where(anti, 0.5 * jnp.exp(lpc - lp[:, None]), 0.0)
        return (0.75 * N_SITES + 2 * sisj @ _PAIR_SIGN.T) / N_SITES**2

    return nk.jax.apply_chunked(batch, chunk_size=CORR_BATCH)(x)


def _order_parameters(vstate, j2, samples):
    """m2_neel and m2_stripe with their standard errors, from the first
    CORR_SAMPLES of a coupling's evaluation samples."""
    chunk = vstate.chunk_size or EVAL_CHUNK
    x = samples[:, : max(1, CORR_SAMPLES // samples.shape[0])]
    c = _local_structure_factors(vstate._apply_fun, vstate.variables, x.reshape(-1, N_SITES), j2, chunk)
    c = c.real.reshape(*x.shape[:-1], 3)
    neel = nk.stats.statistics(c[..., 0])
    stripe = nk.stats.statistics(0.5 * (c[..., 1] + c[..., 2]))
    return (float(neel.mean), float(neel.error_of_mean)), (float(stripe.mean), float(stripe.error_of_mean))


def _v_score(stats):
    """Variational score of Wu et al. (Science 2024) with E_inf = 0: N Var(E) / E^2."""
    return N_SITES * float(stats.variance) / float(stats.mean.real) ** 2


def evaluate(vstate, budget, t_start=None):
    """Score a trained foundation state. Lower `score` is better.

    score = mean over TEST_J2 of log10((E - E0) / |E0|), with E the Monte Carlo
    energy of `vstate` at that coupling; `score_err` is its statistical error.
    """
    t_eval = time.time()
    if isinstance(budget, harness.TrainingBudget) and budget.diverged_at is not None:
        harness.diverged(f"non-finite sampled energy at step {budget.diverged_at}", budget)
    problems = harness.check_run(vstate, budget, BANNED_IN_TRAIN)
    if not problems:
        train_j2 = np.asarray(vstate.parameter_array).ravel()
        dist = np.min(np.abs(train_j2[:, None] - np.array(TEST_J2)[None, :]))
        if dist < MIN_TEST_DISTANCE:
            problems.append(f"a training coupling is {dist:.4f} from a test coupling")
    if problems:
        harness.invalid("; ".join(problems))
    if not harness.check_finite_parameters(vstate):
        harness.diverged("non-finite network parameters after training", budget)

    seed, mc_seed = harness.seeds()
    rel_errs, log_errs, v_scores, r_hats, m2_errs = [], [], [], [], []
    for k, j2 in enumerate(TEST_J2):
        st, samples = _mc_energy(vstate, j2, seed=mc_seed + 1000 * (k + 1))
        e, err, e0 = float(st.mean.real), float(st.error_of_mean), REFERENCE_E0[j2]
        rel = (e - e0) / abs(e0)
        print(f"  J2={j2:.4f}  E/N={e / N_SITES:.8f}({err / N_SITES:.1e})  E0/N={e0 / N_SITES:.8f}  "
              f"rel_err={rel:.3e}  v_score={_v_score(st):.2e}  R_hat={float(st.R_hat):.3f}", flush=True)
        (neel, neel_err), (stripe, stripe_err) = _order_parameters(vstate, j2, samples)
        neel0, stripe0 = REFERENCE_M2[j2]
        m2_errs.append(max(abs(neel - neel0), abs(stripe - stripe0)))
        print(f"            m2_neel={neel:.4f}({neel_err:.4f}) exact {neel0:.4f}  "
              f"m2_stripe={stripe:.4f}({stripe_err:.4f}) exact {stripe0:.4f}", flush=True)
        if not (math.isfinite(e) and math.isfinite(err)):
            harness.diverged(f"non-finite Monte Carlo energy at J2={j2} (amplitudes overflow or NaN)", budget)
        if float(st.R_hat) > MAX_R_HAT:
            harness.invalid(f"evaluation chains did not mix at J2={j2} (R_hat {float(st.R_hat):.3f}): |psi|^2 too hard to sample")
        if e < e0 - BELOW_SIGMAS * err - 1e-9:
            harness.invalid(f"energy {BELOW_SIGMAS:.0f} standard errors below exact ground state at J2={j2} (bug)")
        rel = max(rel, 1e-7)
        rel_errs.append(rel)
        log_errs.append(err / abs(e0) / rel / math.log(10))
        v_scores.append(_v_score(st))
        r_hats.append(float(st.R_hat))
    rel_errs, v_scores = np.array(rel_errs), np.array(v_scores)
    worst = int(np.argmax(rel_errs))

    # Random probe couplings, unknown in advance: a foundation model must be as
    # good between the test points as at them. No exact energy is known there,
    # so they are compared by V-score, which vanishes for an exact eigenstate.
    probes = np.random.default_rng().uniform(TEST_J2[0], TEST_J2[-1], N_PROBES)
    probe_ratio = 0.0
    for k, j2 in enumerate(probes):
        st, _ = _mc_energy(vstate, j2, seed=mc_seed + 100 * (k + 1))
        v = _v_score(st)
        if not math.isfinite(v):
            harness.diverged(f"non-finite Monte Carlo energy at probe J2={j2:.4f}", budget)
        if float(st.R_hat) > MAX_R_HAT:
            harness.invalid(f"evaluation chains did not mix at probe J2={j2:.4f} (R_hat {float(st.R_hat):.3f})")
        i = int(np.searchsorted(TEST_J2, j2))
        ratio = v / v_scores[max(i - 1, 0) : i + 1].max()
        probe_ratio = max(probe_ratio, ratio)
        print(f"  probe J2={j2:.4f}  E/N={float(st.mean.real) / N_SITES:.8f}  v_score={v:.2e}  "
              f"({ratio:.2f}x neighbouring test points)", flush=True)
        if ratio > PROBE_FACTOR:
            harness.invalid(f"{ratio:.0f}x worse V-score at random J2={j2:.4f} than at neighbouring test points (overfit to TEST_J2)")
    harness.summary(
        status="ok",
        score=float(np.mean(np.log10(rel_errs))),
        score_err=float(np.sqrt(np.sum(np.square(log_errs))) / len(log_errs)),
        mean_rel_err=float(np.mean(rel_errs)),
        max_rel_err=float(rel_errs[worst]),
        worst_j2=TEST_J2[worst],
        mean_v_score=float(np.mean(v_scores)),
        max_m2_err=float(max(m2_errs)),
        worst_m2_j2=TEST_J2[int(np.argmax(m2_errs))],
        max_r_hat=float(max(r_hats)),
        probe_ratio=float(probe_ratio),
        training_seconds=float(budget.elapsed),
        eval_seconds=float(time.time() - t_eval),
        total_seconds=float(time.time() - t_start) if t_start else 0.0,
        num_steps=budget.steps,
        num_params=int(vstate.n_parameters),
        peak_vram_gb=float(harness.peak_vram_gb()),
        seed=seed,
        mc_seed=mc_seed,
        device=str(jax.devices()[0].device_kind),
    )


def prepare():
    """Nothing to compute: the reference energies are constants of this file."""
    print(f"J1-J2 {LX}x{LY} PBC, Sz=0, {len(NN_BONDS)} NN + {len(NNN_BONDS)} NNN bonds")
    print(f"Device: {jax.devices()[0].device_kind}")
    for j2 in TEST_J2:
        e0 = REFERENCE_E0[j2]
        print(f"  J2={j2:.4f}  E0={e0:.10f}  E0/N={e0 / N_SITES:.10f}")
