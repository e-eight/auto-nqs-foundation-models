"""
Foundation NQS for the J1-J2 model: the single file the agent edits.

Everything is fair game: model, sampler, training couplings, optimizer,
schedules. The run must train under prepare.TrainingBudget and end with
prepare.evaluate(vstate, budget).

    uv run train.py
"""

import time

t_start = time.time()

import jax.numpy as jnp
import netket as nk
import numpy as np
import netket_foundation as nkf
import optax
from netket_foundation.model import ViTFNQS

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
LEARNING_RATE = 4e-2  # peak of a warmup + cosine decay schedule
LR_INIT, LR_END = 1e-2, 3e-3
WARMUP_STEPS = 50
DECAY_STEPS = 750  # roughly the number of steps that fit in the budget on an H200 slice
DIAG_SHIFT = 1e-3
CHUNK_SIZE = 4096  # forward-pass chunk (local energies)
CHUNK_SIZE_BWD = 512  # backward-pass chunk (SR)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

hi = prepare.hilbert()
ps = prepare.parameter_space()
ham = prepare.hamiltonian(hi, ps)

model = ViTFNQS(
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    heads=HEADS,
    L_eff=hi.size // PATCH,
    n_coups=ps.size,
    b=PATCH,
    complex=True,  # J1-J2 amplitudes are not positive (Marshall sign and beyond)
    disorder=False,
    transl_invariant=True,  # translation along rows-of-the-lattice (PBC in y)
    two_dimensional=False,
)

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
# During training each replica is redrawn every step uniformly within its grid
# cell (stratified sampling of J2), so the whole range is covered; the final
# parameter_array is reset to the grid.
J2_GRID = np.linspace(prepare.J2_MIN, prepare.J2_MAX, N_REPLICAS)
J2_HALF_CELL = 0.5 * (J2_GRID[1] - J2_GRID[0])
vstate.parameter_array = jnp.asarray(J2_GRID).reshape(-1, 1)
rng = np.random.default_rng(SEED)


def resample_couplings(step, log_data, driver):
    j2 = J2_GRID + rng.uniform(-J2_HALF_CELL, J2_HALF_CELL, N_REPLICAS)
    j2 = np.clip(j2, prepare.J2_MIN, prepare.J2_MAX)
    driver.state.parameter_array = jnp.asarray(j2).reshape(-1, 1)
    return True


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
driver.run(10**9, out=log, callback=[resample_couplings, budget], show_progress=False)
vstate.parameter_array = jnp.asarray(J2_GRID).reshape(-1, 1)

energies = log.data["Mean Energy"]["Mean"]
print(f"final mean energy/N (sampled): {float(energies[-1].real) / hi.size:.6f}")
prepare.evaluate(vstate, budget, t_start=t_start)
