"""
Foundation NQS for the J1-J2 model: the single file the agent edits.

Everything is fair game: model, sampler, training couplings, optimizer,
schedules. The run must train under prepare.TrainingBudget and end with
prepare.evaluate(vstate, budget).

    uv run train.py
"""

import time

t_start = time.time()

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import netket_foundation as nkf
import optax

import prepare

# Full float32 matmuls (no TF32): amplitude ratios in the local energy need them.
jax.config.update("jax_default_matmul_precision", "highest")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEED = 0
NUM_LAYERS = 2
D_MODEL = 32
HEADS = 4
PATCH = prepare.LX  # one token per lattice row
N_REPLICAS = 28  # couplings trained simultaneously
CHAINS_PER_REPLICA = 18
SAMPLES_PER_REPLICA = 180
LEARNING_RATE = 6e-2  # peak of a warmup + cosine decay schedule
LR_INIT, LR_END = 1e-2, 3e-3
WARMUP_STEPS = 50
DECAY_STEPS = 750  # roughly the number of steps that fit in the budget on an H200 slice
DIAG_SHIFT = 1e-4
CHUNK_SIZE = 4096  # forward-pass chunk (local energies)
DTYPE = jnp.float32  # network compute dtype (parameters stay float64)
AVERAGE_FRACTION = 0.25  # average the weights over this final fraction of the budget
CHUNK_SIZE_BWD = None  # no backward-pass chunking (SR Jacobian)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

hi = prepare.hilbert()
ps = prepare.parameter_space()
ham = prepare.hamiltonian(hi, ps)

# Symmetries the network does not have built in: translations along x (the
# row-token attention is already invariant under translations along y) and the
# global spin flip, under which the singlet ground state on an even number of
# sites is even. The amplitude is averaged over the group.
X_SHIFTS = np.array(
    [[y * prepare.LX + (x + dx) % prepare.LX for y in range(prepare.LY) for x in range(prepare.LX)]
     for dx in range(prepare.LX)]
)


class Symmetrized(nn.Module):
    net: nn.Module

    def __call__(self, x):
        spins, coups = x[..., : prepare.N_SITES], x[..., prepare.N_SITES :]
        images = jnp.concatenate([spins[..., X_SHIFTS], -spins[..., X_SHIFTS]], axis=-2)
        coups = jnp.broadcast_to(coups[..., None, :], (*images.shape[:-1], coups.shape[-1]))
        lp = self.net(jnp.concatenate([images, coups], axis=-1).reshape(-1, x.shape[-1]))
        lp = lp.reshape(images.shape[:-1])
        m = jnp.max(lp.real, axis=-1, keepdims=True)
        return m[..., 0] + jnp.log(jnp.mean(jnp.exp(lp - m), axis=-1))


SUBLATTICE_A = np.array(
    [(x + y) % 2 == 0 for y in range(prepare.LY) for x in range(prepare.LX)], dtype=np.float64
)


def dense(features):
    return nn.Dense(
        features, kernel_init=nn.initializers.xavier_uniform(), dtype=DTYPE, param_dtype=jnp.float64
    )


def layer_norm():
    return nn.LayerNorm(dtype=DTYPE, param_dtype=jnp.float64)


def log_cosh(x):
    x = x * jnp.where(x.real < 0, -1, 1)
    return x + jnp.log1p(jnp.exp(-2 * x)) - jnp.log(2.0)


class FactoredAttention(nn.Module):
    """Attention with input-independent, translation-invariant weights J[h, j - i]
    across row tokens (the ViT of Viteritti, Rende, Becca, PRL 2023)."""

    @nn.compact
    def __call__(self, x):
        n_tok = x.shape[-2]
        J = self.param("J", nn.initializers.xavier_uniform(), (HEADS, n_tok), jnp.float64)
        J = jnp.stack([jnp.roll(J, i, axis=-1) for i in range(n_tok)], axis=1).astype(DTYPE)  # (h, i, j)
        v = dense(D_MODEL)(x).reshape(*x.shape[:-1], HEADS, D_MODEL // HEADS)
        x = jnp.einsum("hij,...jhd->...ihd", J, v).reshape(*x.shape[:-1], D_MODEL)
        return dense(D_MODEL)(x)


class TwoSignViT(nn.Module):
    """Foundation ViT (couplings concatenated to each row patch, pre-norm encoder
    blocks) whose shared encoder feeds two complex output heads, combined as

        psi = exp(f1) * Marshall(s) + exp(f2),

    so the network can pick the Marshall sign (Neel side) or a learned phase
    (stripe side) per coupling without fractional phases."""

    @staticmethod
    def head(h):
        z = layer_norm()(h)
        return jnp.sum(log_cosh(dense(D_MODEL)(z) + 1j * dense(D_MODEL)(z)), axis=-1).astype(jnp.complex128)

    @nn.compact
    def __call__(self, x):
        spins, coups = x[..., : prepare.N_SITES], x[..., prepare.N_SITES :]
        h = spins.reshape(*spins.shape[:-1], -1, PATCH)
        h = jnp.concatenate([h, jnp.broadcast_to(coups[..., None, :], (*h.shape[:-1], coups.shape[-1]))], -1)
        h = dense(D_MODEL)(h.astype(DTYPE))
        for _ in range(NUM_LAYERS):
            h = h + FactoredAttention()(layer_norm()(h))
            h = h + dense(D_MODEL)(nn.gelu(dense(4 * D_MODEL)(layer_norm()(h))))
        h = h.sum(axis=-2)
        f1, f2 = (self.head(h) for _ in range(2))
        marshall = 1 - 2 * (jnp.sum((1 + spins) / 2 * SUBLATTICE_A, axis=-1) % 2)
        m = jnp.maximum(f1.real, f2.real)
        return m + jnp.log(jnp.exp(f1 - m) * marshall + jnp.exp(f2 - m))


model = Symmetrized(TwoSignViT())

sampler = nk.sampler.MetropolisExchange(
    hi, graph=prepare.graph(), d_max=1, n_chains=N_REPLICAS * CHAINS_PER_REPLICA
)
vstate = nkf.FoundationalQuantumState(
    sampler,
    model,
    ps,
    n_replicas=N_REPLICAS,
    n_samples=N_REPLICAS * SAMPLES_PER_REPLICA,
    seed=SEED,
    chunk_size=CHUNK_SIZE,
)
# Training couplings: a 0.025 grid offset by half a step (0.2125, 0.2375, ...,
# 0.8875), which keeps every point 0.0125 away from the test couplings.
J2_STEP = (prepare.J2_MAX - prepare.J2_MIN) / N_REPLICAS
vstate.parameter_array = (prepare.J2_MIN + J2_STEP * (jnp.arange(N_REPLICAS) + 0.5)).reshape(-1, 1)

driver = nkf.VMC_SR(
    ham,
    optax.sgd(
        optax.warmup_cosine_decay_schedule(
            LR_INIT, LEARNING_RATE, WARMUP_STEPS, DECAY_STEPS, end_value=LR_END
        )
    ),
    variational_state=vstate,
    diag_shift=DIAG_SHIFT,
    use_ntk=True,
    chunk_size_bwd=CHUNK_SIZE_BWD,
    on_the_fly=False,  # explicit Jacobian: ~8x faster than the on-the-fly NTK here
)

print(f"num_params: {vstate.n_parameters}")
print(f"train_j2: {[round(float(p), 4) for p in vstate.parameter_array.ravel()]}")

# ---------------------------------------------------------------------------
# Train for the fixed budget, then evaluate
# ---------------------------------------------------------------------------

budget = prepare.TrainingBudget()


class WeightAverage:
    """Polyak averaging of the parameters over the end of training, to cancel
    the step-to-step noise of the stochastic SR updates."""

    def __init__(self):
        self.mean, self.count = None, 0

    def __call__(self, step, log_data, driver):
        if budget.elapsed >= (1 - AVERAGE_FRACTION) * budget.seconds:
            self.count += 1
            p = driver.state.parameters
            self.mean = p if self.mean is None else jax.tree_util.tree_map(
                lambda m, x: m + (x - m) / self.count, self.mean, p
            )
        return True


average = WeightAverage()
log = nk.logging.RuntimeLog()
driver.run(10**9, out=log, callback=[average, budget], show_progress=False, timeit=True)
print(f"averaged weights over the last {average.count} steps")
if average.count:
    vstate.parameters = average.mean

energies = log.data["Mean Energy"]["Mean"]
print(f"final mean energy/N (sampled): {float(energies[-1].real) / hi.size:.6f}")
prepare.evaluate(vstate, budget, t_start=t_start)
