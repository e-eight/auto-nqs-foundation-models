"""
Fixed problem definition, reference energies and evaluation harness.

DO NOT MODIFY. The agent edits train.py only.

Problem: spin-1/2 J1-J2 Heisenberg model on an LX x LY square lattice with
periodic boundaries, in the total Sz = 0 sector,

    H(J2) = J1 * sum_<ij> S_i . S_j + J2 * sum_<<ij>> S_i . S_j,   J1 = 1,

with spin-1/2 operators (not Pauli matrices). Sites are numbered row-major,
i = y * LX + x, so a row of LX sites is a contiguous block of the input.

A foundation model is trained on J2 in [J2_MIN, J2_MAX] and scored at the
held-out couplings TEST_J2 by *exact* summation over the full Sz = 0 basis,
so the score has no Monte Carlo noise and cannot go below the true
ground-state energy.

Run once per machine to compute the exact reference energies:

    uv run prepare.py
"""

import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import netket_foundation as nkf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LX, LY = 4, 6  # columns, rows (periodic in both directions)
N_SITES = LX * LY
J1 = 1.0
J2_MIN, J2_MAX = 0.2, 0.9  # ParameterSpace range the foundation model covers
TEST_J2 = tuple(round(0.325 + 0.05 * k, 4) for k in range(10))  # 0.325 ... 0.775
MIN_TEST_DISTANCE = 0.01  # final training couplings must stay this far from TEST_J2
N_PROBES = 3  # fresh random couplings per evaluation, to catch overfitting to TEST_J2
PROBE_FACTOR = 10.0  # a probe may be at most this much worse than its neighbouring test points
TIME_BUDGET = 600  # seconds of training, counted from the end of the first step
EVAL_CHUNK = 2**14  # configurations per forward pass during evaluation
CACHE_DIR = Path(os.environ.get("AUTO_NQS_CACHE", "~/.cache/auto-nqs")).expanduser()

# train.py may not use exact or full-basis machinery: training must be VMC.
BANNED_IN_TRAIN = re.compile(
    r"\b(lanczos_ed|eigsh|full_ed|FullSumState|all_states|to_sparse|to_dense|"
    r"TEST_J2|_ExactBasis|_exact_basis|reference_energies)\b"
)

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
    """Spin-1/2 chain of N_SITES sites in the total Sz = 0 sector (values +-1)."""
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
# Exact basis: full Sz = 0 enumeration, sparse H as gather tables on device
# ---------------------------------------------------------------------------


class _ExactBasis:
    def __init__(self):
        assert N_SITES <= 30, "full enumeration only for small lattices"
        allints = np.arange(2**N_SITES, dtype=np.uint32)
        self.states = jnp.asarray(allints[np.bitwise_count(allints) == N_SITES // 2])
        del allints
        self.dim = int(self.states.shape[0])
        bits = jnp.arange(N_SITES, dtype=jnp.uint32)
        self._bits = bits

        def tables(bonds):
            diag = jnp.zeros(self.dim)
            nbrs = []
            for i, j in bonds:
                bi = (self.states >> i) & 1
                bj = (self.states >> j) & 1
                diag = diag + 0.25 * jnp.where(bi == bj, 1.0, -1.0)
                flipped = self.states ^ jnp.uint32((1 << i) | (1 << j))
                idx = jnp.searchsorted(self.states, flipped).astype(jnp.int32)
                nbrs.append(jnp.where(bi != bj, idx, self.dim))  # dim -> zero pad
            return diag, jnp.stack(nbrs)

        self.diag1, self.nbr1 = tables(NN_BONDS)
        self.diag2, self.nbr2 = tables(NNN_BONDS)

    def spins(self, start, size):
        s = jax.lax.dynamic_slice(self.states, (start,), (size,))
        return 2.0 * ((s[:, None] >> self._bits) & 1).astype(jnp.float64) - 1.0

    @staticmethod
    @jax.jit
    def _offdiag(v, nbr):
        v_ext = jnp.concatenate([v, jnp.zeros(1, v.dtype)])
        acc, _ = jax.lax.scan(lambda a, n: (a + v_ext[n], None), jnp.zeros_like(v), nbr)
        return 0.5 * acc

    def matvec(self, v, j2):
        h1 = self.diag1 * v + self._offdiag(v, self.nbr1)
        h2 = self.diag2 * v + self._offdiag(v, self.nbr2)
        return J1 * h1 + j2 * h2

    def ground_energy(self, j2):
        from scipy.sparse.linalg import LinearOperator, eigsh

        op = LinearOperator(
            (self.dim, self.dim),
            matvec=lambda v: np.asarray(self.matvec(jnp.asarray(v.ravel()), j2)),
            dtype=np.float64,
        )
        v0 = np.random.default_rng(0).standard_normal(self.dim)
        return float(eigsh(op, k=1, which="SA", v0=v0, tol=1e-12)[0][0])


_BASIS = None


def _exact_basis():
    global _BASIS
    if _BASIS is None:
        _BASIS = _ExactBasis()
    return _BASIS


# ---------------------------------------------------------------------------
# Reference energies (cached on disk)
# ---------------------------------------------------------------------------


def _ref_path():
    key = json.dumps([LX, LY, J1, list(TEST_J2)])
    return CACHE_DIR / f"refs-{hashlib.sha1(key.encode()).hexdigest()[:12]}.json"


def reference_energies():
    """Exact ground-state energies at TEST_J2, computed once and cached."""
    path = _ref_path()
    if path.exists():
        return {float(k): v for k, v in json.loads(path.read_text()).items()}
    basis = _exact_basis()
    refs = {}
    for j2 in TEST_J2:
        t = time.time()
        refs[j2] = basis.ground_energy(j2)
        print(f"  J2={j2:.4f}  E0={refs[j2]:.10f}  E0/N={refs[j2] / N_SITES:.10f}  ({time.time() - t:.1f}s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in refs.items()}, indent=1))
    return refs


# ---------------------------------------------------------------------------
# Training budget
# ---------------------------------------------------------------------------


class TrainingBudget:
    """NetKet driver callback that stops training after TIME_BUDGET seconds.

    The clock starts when the first optimisation step finishes, so the first
    JIT compilation is not counted. Pass it as `driver.run(..., callback=budget)`
    and hand the same object to `evaluate`.
    """

    def __init__(self):
        self.seconds = TIME_BUDGET
        self.t0 = None
        self.elapsed = 0.0
        self.last_step = 0.0
        self.steps = 0

    def __call__(self, step, log_data, driver):
        now = time.time()
        if self.t0 is None:
            self.t0 = now
            return True
        self.steps += 1
        self.last_step = (now - self.t0) - self.elapsed
        self.elapsed = now - self.t0
        return self.elapsed < self.seconds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _exact_energy(vstate, basis, j2):
    chunk = min(EVAL_CHUNK, basis.dim)
    logpsi = []
    for c in range(math.ceil(basis.dim / chunk)):
        # Fixed chunk shape (one compilation); the last chunk overlaps the previous one.
        start = min(c * chunk, basis.dim - chunk)
        x = basis.spins(start, chunk)
        x = jnp.concatenate([x, jnp.full((chunk, 1), j2)], axis=-1)
        lp = vstate.log_value(x)
        logpsi.append(lp[c * chunk - start :])
    logpsi = jnp.concatenate(logpsi)
    v = jnp.exp(logpsi - jnp.max(logpsi.real))
    hv = basis.matvec(v, j2)
    return float(jnp.vdot(v, hv).real / jnp.vdot(v, v).real)


def _peak_vram_gb():
    try:
        stats = jax.devices()[0].memory_stats() or {}
        return stats.get("peak_bytes_in_use", 0) / 1024**3
    except Exception:
        return 0.0


def _summary(**fields):
    print("---")
    for k, v in fields.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    sys.stdout.flush()


def evaluate(vstate, budget, t_start=None):
    """Score a trained foundation state. Lower `score` is better.

    score = mean over TEST_J2 of log10((E - E0) / |E0|), with E the exact
    variational energy of `vstate` at that coupling.
    """
    t_eval = time.time()
    problems = []
    train_src = Path(sys.argv[0]).read_text() if Path(sys.argv[0]).name == "train.py" else ""
    hit = BANNED_IN_TRAIN.search(train_src)
    if hit:
        problems.append(f"train.py uses banned name '{hit.group(0)}'")
    if not isinstance(vstate, nkf.FoundationalQuantumState):
        problems.append("vstate is not a netket_foundation.FoundationalQuantumState")
    if not isinstance(budget, TrainingBudget) or budget.t0 is None:
        problems.append("training did not run under prepare.TrainingBudget")
    elif budget.elapsed - budget.last_step >= TIME_BUDGET:
        problems.append(f"training continued past the {TIME_BUDGET}s budget ({budget.elapsed:.0f}s)")
    if not problems:
        train_j2 = np.asarray(vstate.parameter_array).ravel()
        dist = np.min(np.abs(train_j2[:, None] - np.array(TEST_J2)[None, :]))
        if dist < MIN_TEST_DISTANCE:
            problems.append(f"a training coupling is {dist:.4f} from a test coupling")
    if problems:
        _summary(status="invalid", reason="; ".join(problems))
        sys.exit(2)

    refs = reference_energies()
    basis = _exact_basis()
    rel_errs = []
    for j2 in TEST_J2:
        e = _exact_energy(vstate, basis, j2)
        rel = (e - refs[j2]) / abs(refs[j2])
        print(f"  J2={j2:.4f}  E/N={e / N_SITES:.8f}  E0/N={refs[j2] / N_SITES:.8f}  rel_err={rel:.3e}", flush=True)
        if rel < -1e-9:
            _summary(status="invalid", reason=f"energy below exact ground state at J2={j2} (bug)")
            sys.exit(2)
        rel_errs.append(max(rel, 1e-16))
    rel_errs = np.array(rel_errs)
    worst = int(np.argmax(rel_errs))

    # Random probe couplings, unknown in advance: a foundation model must be as
    # good between the test points as at them.
    probes = np.random.default_rng().uniform(TEST_J2[0], TEST_J2[-1], N_PROBES)
    probe_ratio = 0.0
    for j2 in probes:
        e0 = basis.ground_energy(j2)
        rel = max((_exact_energy(vstate, basis, j2) - e0) / abs(e0), 1e-16)
        k = int(np.searchsorted(TEST_J2, j2))
        neighbours = rel_errs[max(k - 1, 0) : k + 1].max()
        ratio = rel / neighbours
        probe_ratio = max(probe_ratio, ratio)
        print(f"  probe J2={j2:.4f}  rel_err={rel:.3e}  ({ratio:.2f}x neighbouring test points)", flush=True)
        if ratio > PROBE_FACTOR and rel > 1e-6:
            _summary(status="invalid", reason=f"{ratio:.0f}x worse at random J2={j2:.4f} than at neighbouring test points (overfit to TEST_J2)")
            sys.exit(2)
    _summary(
        status="ok",
        score=float(np.mean(np.log10(rel_errs))),
        mean_rel_err=float(np.mean(rel_errs)),
        max_rel_err=float(rel_errs[worst]),
        worst_j2=TEST_J2[worst],
        probe_ratio=float(probe_ratio),
        training_seconds=float(budget.elapsed),
        eval_seconds=float(time.time() - t_eval),
        total_seconds=float(time.time() - t_start) if t_start else 0.0,
        num_steps=budget.steps,
        num_params=int(vstate.n_parameters),
        peak_vram_gb=float(_peak_vram_gb()),
        device=str(jax.devices()[0].device_kind),
    )


if __name__ == "__main__":
    print(f"J1-J2 {LX}x{LY} PBC, Sz=0, {len(NN_BONDS)} NN + {len(NNN_BONDS)} NNN bonds")
    print(f"Device: {jax.devices()[0].device_kind}")
    t = time.time()
    basis = _exact_basis()
    print(f"Sz=0 basis: {basis.dim} states ({time.time() - t:.1f}s)")
    reference_energies()
    print(f"References cached at {_ref_path()}")
