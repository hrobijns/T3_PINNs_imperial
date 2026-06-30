"""
verification/verify_flat_metric.py — Verify PINN learns constant harmonic forms on T^3.

For the flat metric g=I, harmonic 1-forms are exactly constant (Fourier argument).
The PINN should learn a nearly-constant non-zero form with low PDE loss.

Pass criteria:
  1. Coefficient of variation of ||lambda(x)|| < 0.10  (nearly constant)
  2. min_x ||lambda(x)|| > 0.05  (no zeros on flat torus)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from core.metrics import FlatMetric
from core.networks import build_network
from core.differential_geometry import DiffGeomOps

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name, cond, detail=""):
    print(f"  [{ PASS if cond else FAIL }] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def verify_flat_metric(epochs: int = 500, n_collocations: int = 2000,
                       n_eval: int = 500, print_every: int = 50) -> bool:
    """Train a PINN on the flat metric and check it learns a constant harmonic form."""
    print("\n" + "=" * 60)
    print("Flat metric verification (g = I_3)")
    print("=" * 60)

    np.random.seed(1); tf.random.set_seed(1)
    metric = FlatMetric()
    ops    = DiffGeomOps(metric)
    model  = build_network()

    x_col = tf.constant(np.random.uniform(0, 1, (n_collocations, 3)), dtype=tf.float64)

    # PDE loss for single PINN (mirroring the original PINN.pde_error logic)
    def pde_error(x_point):
        x = tf.expand_dims(x_point, 0)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            lam      = model(x)
            star_lam = ops.star_1form(lam, x)
        d_lam    = ops.exterior_derivative_1_form(tape, lam, x)
        d_star   = ops.exterior_derivative_2_form(tape, star_lam, x)
        delta    = ops.star_3form(d_star, x)
        del tape
        return tf.reduce_sum(tf.square(d_lam)) + tf.reduce_sum(tf.square(delta))

    def loss_fn(x_col):
        errs      = tf.vectorized_map(pde_error, x_col)
        pde_mean  = tf.reduce_mean(errs)
        u_vals    = model(x_col)
        norm_term = tf.reduce_mean(tf.square(u_vals))
        return pde_mean + 1e2 * tf.square(norm_term - 1.0)

    @tf.function
    def train_step(x_col):
        with tf.GradientTape() as tape:
            L = loss_fn(x_col)
        grads = tape.gradient(L, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return L

    opt = tf.keras.optimizers.Adam(1e-3)
    for ep in range(epochs):
        L = train_step(x_col)
        if ep % print_every == 0:
            print(f"  Epoch {ep:4d} | Loss {L.numpy():.6e}")

    # Evaluate
    x_eval = tf.constant(np.random.uniform(0, 1, (n_eval, 3)), dtype=tf.float64)
    u      = model(x_eval).numpy()
    norms  = np.linalg.norm(u, axis=1)

    mean_norm = float(np.mean(norms))
    std_norm  = float(np.std(norms))
    min_norm  = float(np.min(norms))
    cv        = std_norm / (mean_norm + 1e-12)

    print(f"\n  mean ||lambda|| = {mean_norm:.4f}")
    print(f"  std  ||lambda|| = {std_norm:.4f}  (CV = {cv:.4f})")
    print(f"  min  ||lambda|| = {min_norm:.4f}")

    ok_cv  = check("Form nearly constant (CV < 0.10)",    cv       < 0.10, f"CV={cv:.4f}")
    ok_min = check("No zeros on flat torus (min > 0.05)", min_norm > 0.05, f"min={min_norm:.4f}")
    return ok_cv and ok_min


if __name__ == '__main__':
    ok = verify_flat_metric(epochs=500, print_every=50)
    import sys
    sys.exit(0 if ok else 1)
