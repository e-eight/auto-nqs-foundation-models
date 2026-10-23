"""
Shared harness for every task: seeds, training budget, divergence guard and the
summary block that remote.py parses.

DO NOT MODIFY. The agent edits tasks/<task>/train.py only.

Exit codes of a run: 0 ok, 2 invalid, 3 diverged (anything else is a crash).
"""

import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import jax

EXIT_INVALID = 2
EXIT_DIVERGED = 3


def seeds():
    """(init_seed, mc_seed) of this run, set by `remote.py run --seeds/--vary`.

    The init seed draws the network weights; the Monte Carlo seed drives the
    sampler. Keeping them separate lets us tell which one a result depends on.
    """
    init = int(os.environ.get("NQS_SEED", 0))
    mc = int(os.environ.get("NQS_MC_SEED", init + 10_000))
    return init, mc


def _loss_is_finite(log_data):
    for key in ("Mean Energy", "Energy"):
        stats = log_data.get(key) if hasattr(log_data, "get") else None
        if stats is not None:
            mean = getattr(stats, "mean", stats)
            return bool(np.all(np.isfinite(np.asarray(mean))))
    return True


class TrainingBudget:
    """NetKet driver callback that stops training after `seconds` of wall clock,
    or as soon as the sampled energy is NaN or infinite.

    The clock starts when the first optimisation step finishes, so the first
    JIT compilation is not counted. Pass it as `driver.run(..., callback=budget)`
    and hand the same object to the task's `evaluate`.
    """

    def __init__(self, seconds):
        self.seconds = seconds
        self.t0 = None
        self.elapsed = 0.0
        self.last_step = 0.0
        self.steps = 0
        self.diverged_at = None

    def __call__(self, step, log_data, driver):
        if not _loss_is_finite(log_data):
            self.diverged_at = self.steps + 1
            print(f"non-finite energy at step {self.diverged_at}; stopping", flush=True)
            return False
        now = time.time()
        if self.t0 is None:
            self.t0 = now
            return True
        self.steps += 1
        self.last_step = (now - self.t0) - self.elapsed
        self.elapsed = now - self.t0
        return self.elapsed < self.seconds


def check_run(vstate, budget, banned):
    """Rule checks shared by all tasks; returns a list of problems."""
    import netket_foundation as nkf

    problems = []
    train = Path(sys.argv[0])
    hit = banned.search(train.read_text()) if train.name == "train.py" else None
    if hit:
        problems.append(f"train.py uses banned name '{hit.group(0)}'")
    if not isinstance(vstate, nkf.FoundationalQuantumState):
        problems.append("vstate is not a netket_foundation.FoundationalQuantumState")
    if not isinstance(budget, TrainingBudget) or budget.t0 is None:
        problems.append("training did not run under harness.TrainingBudget")
    elif budget.elapsed - budget.last_step >= budget.seconds:
        problems.append(f"training continued past the {budget.seconds}s budget ({budget.elapsed:.0f}s)")
    return problems


def check_finite_parameters(vstate):
    leaves = jax.tree_util.tree_leaves(vstate.parameters)
    return all(bool(jax.numpy.all(jax.numpy.isfinite(x))) for x in leaves)


def peak_vram_gb():
    try:
        stats = jax.devices()[0].memory_stats() or {}
        return stats.get("peak_bytes_in_use", 0) / 1024**3
    except Exception:
        return 0.0


def summary(**fields):
    print("---")
    for k, v in fields.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
    sys.stdout.flush()


def invalid(reason):
    summary(status="invalid", reason=reason)
    sys.exit(EXIT_INVALID)


def diverged(reason, budget=None):
    fields = {"status": "diverged", "reason": reason}
    if budget is not None:
        fields["num_steps"] = budget.steps
    summary(**fields)
    sys.exit(EXIT_DIVERGED)


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


BANNED_COMMON = r"lanczos_ed|eigsh|full_ed|FullSumState|all_states|to_sparse|to_dense|reference_energies"


def banned(*task_names):
    """Regex of names train.py may not use: exact machinery plus task-specific ones."""
    return re.compile(r"\b(" + "|".join([BANNED_COMMON, *task_names]) + r")\b")
