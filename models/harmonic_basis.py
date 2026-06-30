"""
models/harmonic_basis.py — HarmonicBasisFinder and metric search loop.

Learns three linearly-independent harmonic 1-forms on T^3 simultaneously.
Loss = PDE residual + norm penalty + orthogonality penalty.
"""
import gc
from typing import Optional

import numpy as np
import tensorflow as tf

from core.differential_geometry import DiffGeomOps
from core.metrics import InputDependentRandomMetric
from core.networks import build_network
from models.zero_finder import find_form_zero

tf.keras.backend.set_floatx('float64')


class HarmonicBasisFinder:
    """
    Learns THREE linearly-independent harmonic 1-forms on T^3 for a given metric.

    Loss = sum_i pde_loss(lambda_i)
         + W_NORM  * sum_i (mean_g_norm_sq(lambda_i) - 1)^2
         + W_ORTHO * sum_{i<j} <lambda_i, lambda_j>^2
    """

    W_NORM  = 100.0
    W_ORTHO = 500.0

    def __init__(self, metric_provider, max_freq: int = 1, width: int = 64):
        self.metric_provider = metric_provider
        self.models = [build_network(max_freq=max_freq, width=width)
                       for _ in range(3)]
        self._ops   = DiffGeomOps(metric_provider)
        # Optimizers and variable lists created lazily on first train() call,
        # then reused across trials to avoid tf.function retracing.
        self._opts      = None
        self._var_lists = None
        self._sizes     = None
        self._all_vars  = None

    # ── Metric passthrough ──────────────────────────────────────────────────
    def metric_tensor(self, x):
        return self._ops.metric_tensor(x)

    # ── Metric / weight reseeding ────────────────────────────────────────────
    def reseed_metric(self, seed: int) -> None:
        """Change metric parameters in-place — no tf.function retracing."""
        self.metric_provider.reseed(seed)

    def reset_weights(self) -> None:
        """Re-initialise model weights in-place (same tf.Variables, new values)."""
        glorot = tf.keras.initializers.GlorotUniform()
        for model in self.models:
            new_weights = []
            for w in model.weights:
                if len(w.shape) == 1:
                    new_weights.append(np.zeros(w.shape, dtype=np.float64))
                else:
                    new_weights.append(glorot(w.shape, dtype=tf.float64).numpy())
            model.set_weights(new_weights)

    def _reset_optimizers(self) -> None:
        """Zero Adam slot variables and iteration counter."""
        for opt in self._opts:
            try:
                opt_vars = opt.variables()
            except TypeError:
                opt_vars = opt.variables
            for var in opt_vars:
                var.assign(tf.zeros_like(var))

    # ── PDE residual ─────────────────────────────────────────────────────────
    def _pde_error_batch(self, model, x_col: tf.Tensor) -> tf.Tensor:
        """||dλ||² + ||δλ||² averaged over the batch for one model."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_col)
            lam      = model(x_col)
            star_lam = self._ops.star_1form(lam, x_col)
        d_lam  = self._ops.exterior_derivative_1_form(tape, lam, x_col)
        d_star = self._ops.exterior_derivative_2_form(tape, star_lam, x_col)
        delta  = self._ops.star_3form(d_star, x_col)
        del tape
        return tf.reduce_mean(
            tf.reduce_sum(tf.square(d_lam), axis=1)
            + tf.square(tf.squeeze(delta, axis=1))
        )

    # ── Loss ─────────────────────────────────────────────────────────────────
    def loss(self, x_col: tf.Tensor) -> tf.Tensor:
        g_inv   = tf.linalg.inv(self.metric_tensor(x_col))
        lambdas = [m(x_col) for m in self.models]

        pde = sum(self._pde_error_batch(m, x_col) for m in self.models)

        norm_penalty = self.W_NORM * sum(
            tf.square(DiffGeomOps.l2_inner(lam, lam, g_inv) - 1.0)
            for lam in lambdas
        )
        ortho = self.W_ORTHO * sum(
            tf.square(DiffGeomOps.l2_inner(lambdas[i], lambdas[j], g_inv))
            for i in range(3) for j in range(i + 1, 3)
        )
        return pde + norm_penalty + ortho

    # ── Training ──────────────────────────────────────────────────────────────
    @tf.function(reduce_retracing=True)
    def _train_step(self, x_col: tf.Tensor) -> tf.Tensor:
        """Single gradient step — compiled once per HarmonicBasisFinder instance."""
        with tf.GradientTape() as tape:
            L = self.loss(x_col)
        all_grads = tape.gradient(L, self._all_vars)
        offset = 0
        for opt, vl, sz in zip(self._opts, self._var_lists, self._sizes):
            opt.apply_gradients(zip(all_grads[offset:offset + sz], vl))
            offset += sz
        return L

    def train(self, x_col: tf.Tensor, epochs: int = 500,
              lr: float = 1e-3, print_every: int = 100,
              clipnorm: float = None, lr_decay: bool = False):
        if self._opts is None:
            if lr_decay:
                schedule = tf.keras.optimizers.schedules.CosineDecay(
                    lr, decay_steps=epochs, alpha=1e-5 / lr)
                self._opts = [tf.keras.optimizers.Adam(schedule, clipnorm=clipnorm)
                              for _ in self.models]
            else:
                self._opts = [tf.keras.optimizers.Adam(lr, clipnorm=clipnorm)
                              for _ in self.models]
            self._var_lists = [m.trainable_variables for m in self.models]
            self._sizes     = [len(v) for v in self._var_lists]
            self._all_vars  = [v for vl in self._var_lists for v in vl]
        else:
            self._reset_optimizers()

        for ep in range(epochs):
            L = self._train_step(x_col)
            if ep % print_every == 0:
                print(f"    Epoch {ep:4d} | Loss {L.numpy():.4e}")

    # ── Held-out evaluation ────────────────────────────────────────────────────
    def evaluate(self, x_eval: tf.Tensor) -> dict:
        """
        Evaluate learned forms on a held-out grid.

        Returns dict with per-form:
          pde_losses   : list[float]  — PDE residual on held-out points
          min_norms    : list[float]  — min pointwise ||λ_i(x)|| over grid
          max_norms    : list[float]  — max pointwise ||λ_i(x)||
          mean_norms   : list[float]  — mean pointwise ||λ_i(x)||
          gram_matrix  : 3x3 list     — L² Gram matrix ⟨λ_i, λ_j⟩_g
          trivial_flags: list[bool]   — True if form looks near-constant
        """
        g_inv = tf.linalg.inv(self.metric_tensor(x_eval))
        lambdas = [m(x_eval) for m in self.models]

        pde_losses = [float(self._pde_error_batch(m, x_eval).numpy())
                      for m in self.models]

        pointwise_norms = []
        for lam in lambdas:
            norms = tf.sqrt(tf.reduce_sum(tf.square(lam), axis=1)).numpy()
            pointwise_norms.append(norms)

        min_norms  = [float(n.min()) for n in pointwise_norms]
        max_norms  = [float(n.max()) for n in pointwise_norms]
        mean_norms = [float(n.mean()) for n in pointwise_norms]

        trivial_flags = []
        for mn, mx in zip(min_norms, max_norms):
            ratio = mx / mn if mn > 1e-10 else float('inf')
            trivial_flags.append(ratio < 5.0)

        gram = [[float(DiffGeomOps.l2_inner(lambdas[i], lambdas[j], g_inv))
                 for j in range(3)] for i in range(3)]

        return {
            'pde_losses':    pde_losses,
            'min_norms':     min_norms,
            'max_norms':     max_norms,
            'mean_norms':    mean_norms,
            'gram_matrix':   gram,
            'trivial_flags': trivial_flags,
        }

    # ── Zero finding ─────────────────────────────────────────────────────────
    def find_zero(self, form_idx: int, tol: float = 1e-3, **kwargs):
        """Find a zero of form_idx on T^3.  Returns (found, min_norm, best_x)."""
        return find_form_zero(self.models[form_idx], tol=tol, **kwargs)

    def verify_zero(self, form_idx: int, x_zero: np.ndarray,
                    radius: float = 0.01, n_neighbors: int = 50) -> float:
        """Check PDE residual near a candidate zero. Returns mean PDE residual."""
        center = x_zero[None, :]
        offsets = np.random.uniform(-radius, radius, size=(n_neighbors, 3))
        pts = np.clip(center + offsets, 0.0, 1.0)
        pts = np.vstack([center, pts])
        x_check = tf.constant(pts, dtype=tf.float64)
        return float(self._pde_error_batch(self.models[form_idx], x_check).numpy())

    @staticmethod
    def _l2_inner(a: tf.Tensor, b: tf.Tensor, g_inv: tf.Tensor) -> tf.Tensor:
        """L^2 inner product: mean_x a^T g^{-1} b. Kept for backward compatibility."""
        return DiffGeomOps.l2_inner(a, b, g_inv)


# ── Metric search ─────────────────────────────────────────────────────────────

def search_for_metric_with_zeros(
        n_collocations: int   = 2000,
        n_trials:       int   = 150,
        epochs:         int   = 2000,
        lr:             float = 1e-3,
        tol:            float = 1e-3,
        pde_tol:        float = 0.05,
        print_every:    int   = 200,
        metric_class          = None,
        net_max_freq:   int   = 1,
        net_width:      int   = 64,
        on_trial_end          = None,
        **metric_kwargs,
) -> dict:
    """
    Search for a spatially-varying metric on T^3 where each of the 3 basis
    harmonic 1-forms has a zero.

    metric_class: class with reseed(seed) and metric_tensor(x) — defaults to
                  InputDependentRandomMetric.  Pass extra constructor kwargs via
                  **metric_kwargs (e.g. max_freq=2 for FourierMetric).
    on_trial_end: optional callable(log: dict) invoked after each trial's log
                  is finalised — use for crash-safe incremental result saving.

    Returns a dict with keys: found (bool), seed (int or None),
    results (list), trial_logs (list of per-trial dicts).
    """
    if metric_class is None:
        metric_class = InputDependentRandomMetric

    print("\n" + "=" * 60)
    print(f"Searching for metric with harmonic zeros  [{metric_class.__name__}]")
    print("=" * 60)

    np.random.seed(0); tf.random.set_seed(0)
    x_col = tf.convert_to_tensor(
        np.random.uniform(0, 1, size=(n_collocations, 3)), dtype=tf.float64)

    # Held-out evaluation grid (separate from training collocation)
    rng_eval = np.random.RandomState(9999)
    x_eval = tf.convert_to_tensor(
        rng_eval.uniform(0, 1, size=(10000, 3)), dtype=tf.float64)

    print(f"  network: max_freq={net_max_freq}, width={net_width}")
    basis = HarmonicBasisFinder(metric_class(seed=0, **metric_kwargs),
                                max_freq=net_max_freq, width=net_width)
    trial_logs = []

    for seed in range(n_trials):
        print(f"\n  seed={seed:3d} (spatially varying)")
        basis.reseed_metric(seed)
        basis.reset_weights()
        basis.train(x_col, epochs=epochs, lr=lr, print_every=print_every,
                    clipnorm=1.0, lr_decay=True)

        # Held-out evaluation
        eval_result = basis.evaluate(x_eval)
        pde_losses = eval_result['pde_losses']
        log = {'seed': seed, 'pde_losses': pde_losses,
               'pde_losses_heldout': pde_losses,
               'eval_min_norms': eval_result['min_norms'],
               'eval_max_norms': eval_result['max_norms'],
               'eval_mean_norms': eval_result['mean_norms'],
               'trivial_flags': eval_result['trivial_flags'],
               'gram_matrix': eval_result['gram_matrix']}

        print(f"  held-out PDE: [{pde_losses[0]:.3e}, {pde_losses[1]:.3e}, {pde_losses[2]:.3e}]")
        print(f"  eval norms (min/mean/max): "
              f"[{eval_result['min_norms'][0]:.3f}/{eval_result['mean_norms'][0]:.3f}/{eval_result['max_norms'][0]:.3f}, "
              f"{eval_result['min_norms'][1]:.3f}/{eval_result['mean_norms'][1]:.3f}/{eval_result['max_norms'][1]:.3f}, "
              f"{eval_result['min_norms'][2]:.3f}/{eval_result['mean_norms'][2]:.3f}/{eval_result['max_norms'][2]:.3f}]")

        if any(eval_result['trivial_flags']):
            trivial_idx = [i for i, f in enumerate(eval_result['trivial_flags']) if f]
            print(f"  WARNING: forms {trivial_idx} look near-constant (max/min ratio < 5)")

        if max(pde_losses) > pde_tol:
            print(f"  NOT CONVERGED — skipping zero check")
            log['converged'] = False
            trial_logs.append(log)
            if on_trial_end is not None:
                on_trial_end(log)
            continue

        log['converged'] = True
        results   = [basis.find_zero(i, tol=tol) for i in range(3)]
        min_norms = [r[1] for r in results]

        # Verify PDE residual at found zeros
        zero_pde_checks = []
        for i, (found_i, mn_i, pt_i) in enumerate(results):
            if found_i:
                local_pde = basis.verify_zero(i, pt_i)
                zero_pde_checks.append(local_pde)
                if local_pde > pde_tol:
                    print(f"  REJECTED zero for λ_{i+1}: PDE residual at zero = {local_pde:.3e} > {pde_tol}")
                    results[i] = (False, mn_i, pt_i)
            else:
                zero_pde_checks.append(None)

        # Check for trivial forms claiming zeros
        for i, (found_i, mn_i, pt_i) in enumerate(results):
            if found_i and eval_result['trivial_flags'][i]:
                print(f"  REJECTED zero for λ_{i+1}: form is near-constant (trivial)")
                results[i] = (False, mn_i, pt_i)

        min_norms = [r[1] for r in results]
        found_all = all(r[0] for r in results)
        log['min_norms']       = min_norms
        log['found_all']       = found_all
        log['zero_locs']       = [r[2].tolist() for r in results]
        log['zero_pde_checks'] = zero_pde_checks
        log['n_zeros_found']   = sum(1 for r in results if r[0])
        trial_logs.append(log)
        if on_trial_end is not None:
            on_trial_end(log)
        n_found = log['n_zeros_found']
        status = "ALL ZEROS FOUND" if found_all else f"{n_found}/3 zeros"
        print(f"  min norms: [{min_norms[0]:.4f}, {min_norms[1]:.4f}, {min_norms[2]:.4f}]  => {status}")

        if found_all:
            print(f"\n*** Metric seed={seed} has zeros for all 3 harmonic forms!")
            for i, (_, mn, pt) in enumerate(results):
                print(f"  lambda_{i+1} at x={np.round(pt, 4)},  ||lambda||={mn:.2e}")
            return {'found': True, 'seed': seed,
                    'results': results, 'trial_logs': trial_logs}

    # Report near misses
    near_misses = [t for t in trial_logs if t.get('n_zeros_found', 0) == 2]
    if near_misses:
        print(f"\nNear misses (2/3 zeros): seeds {[t['seed'] for t in near_misses]}")

    print(f"\nNo metric found after {n_trials} trials.")
    return {'found': False, 'seed': None, 'results': [], 'trial_logs': trial_logs}
