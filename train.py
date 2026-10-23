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
import jax.numpy as jnp
import netket as nk
import numpy as np
import netket_foundation as nkf
import optax
from netket_foundation._src.model.vit import Encoder, OutputHead, extract_patches1d

import prepare

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

SEED = 0
NUM_LAYERS = 2
D_MODEL = 32
HEADS = 4
PATCH = prepare.LX  # one token per lattice row
N_REPLICAS = 15  # couplings trained simultaneously
CHAINS_PER_REPLICA = 34
SAMPLES_PER_REPLICA = 272
LEARNING_RATE = 6e-2  # peak of a warmup + cosine decay schedule
LR_INIT, LR_END = 1e-2, 3e-3
WARMUP_STEPS = 50
DECAY_STEPS = 750  # roughly the number of steps that fit in the budget on an H200 slice
DIAG_SHIFT = 1e-4
CHUNK_SIZE = 4096  # forward-pass chunk (local energies)
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


class TwoSignViT(nn.Module):
    """Foundation ViT (couplings concatenated to each row patch) whose shared
    encoder feeds two complex output heads, combined as

        psi = exp(f1) * Marshall(s) + exp(f2),

    so the network can pick the Marshall sign (Neel side) or a learned phase
    (stripe side) per coupling without fractional phases."""

    @nn.compact
    def __call__(self, x):
        spins, coups = x[..., : prepare.N_SITES], x[..., prepare.N_SITES :]
        h = extract_patches1d(spins, PATCH)
        h = jnp.concatenate([h, jnp.broadcast_to(coups[..., None, :], (*h.shape[:-1], coups.shape[-1]))], -1)
        h = nn.Dense(D_MODEL, kernel_init=nn.initializers.xavier_uniform(), param_dtype=jnp.float64)(h)
        h = Encoder(NUM_LAYERS, D_MODEL, HEADS, prepare.N_SITES // PATCH, transl_invariant=True)(h)
        f1 = OutputHead(D_MODEL, complex=True)(h)
        f2 = OutputHead(D_MODEL, complex=True)(h)
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
# Training couplings: multiples of 0.05 in [J2_MIN, J2_MAX], between the test points.
vstate.parameter_array = jnp.linspace(prepare.J2_MIN, prepare.J2_MAX, N_REPLICAS).reshape(-1, 1)

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
log = nk.logging.RuntimeLog()
driver.run(10**9, out=log, callback=budget, show_progress=False, timeit=True)

energies = log.data["Mean Energy"]["Mean"]
print(f"final mean energy/N (sampled): {float(energies[-1].real) / hi.size:.6f}")
prepare.evaluate(vstate, budget, t_start=t_start)
