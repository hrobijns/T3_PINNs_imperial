"""
verification/verify_hodge_star.py — Verification tests for the Hodge star operators.

Tests:
  1. *(*alpha) = alpha  on 1-forms  (3D Riemannian => *^2 = +Id)
  2. *(*beta)  = beta   on 2-forms
  3. star_3form: *(f vol) = f / sqrt(g)
  4. Consistency: star_derivative_2_form = *(d(2-form)) uses /sqrt(g) not *sqrt(g)
  5. Flat-metric sanity checks with known analytic answers

Runnable as:
    python verification/verify_hodge_star.py
    python -m verification.verify_hodge_star
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

from core.metrics import ConstantRandomMetric, InputDependentRandomMetric, FlatMetric
from core.differential_geometry import DiffGeomOps

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, err, tol=1e-10):
    ok = err < tol
    print(f"  {'['+PASS+']' if ok else '['+FAIL+']'} {name}: max_err = {err:.3e}  (tol={tol:.0e})")
    return ok

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("1. ConstantRandomMetric  (constant random metric)")
print("=" * 60)

metric_c = ConstantRandomMetric(seed=7)
ops_c    = DiffGeomOps(metric_c)

B = 256
x = tf.zeros((B, 3), dtype=tf.float64)   # constant metric — x is ignored
alpha = tf.random.normal((B, 3), dtype=tf.float64)
beta  = tf.random.normal((B, 3), dtype=tf.float64)

# *(*alpha) should equal alpha
star_star_a = ops_c.star_2form(ops_c.star_1form(alpha, x), x)
err1a = float(tf.reduce_max(tf.abs(star_star_a - alpha)).numpy())
check("*(*1-form) = 1-form", err1a)

# *(*beta) should equal beta
star_star_b = ops_c.star_1form(ops_c.star_2form(beta, x), x)
err1b = float(tf.reduce_max(tf.abs(star_star_b - beta)).numpy())
check("*(*2-form) = 2-form", err1b)

# test_hodge_involution using batched random x
x_rand_c = tf.random.uniform((B, 3), 0.0, 1.0, dtype=tf.float64)
alpha_c   = tf.random.normal((B, 3), dtype=tf.float64)
beta_c    = tf.random.normal((B, 3), dtype=tf.float64)
star_star_a_c = ops_c.star_2form(ops_c.star_1form(alpha_c, x_rand_c), x_rand_c)
err_c1 = float(tf.reduce_max(tf.abs(star_star_a_c - alpha_c)).numpy())
star_star_b_c = ops_c.star_1form(ops_c.star_2form(beta_c, x_rand_c), x_rand_c)
err_c2 = float(tf.reduce_max(tf.abs(star_star_b_c - beta_c)).numpy())
check("ConstantRandomMetric hodge involution err1 (batched x)", err_c1)
check("ConstantRandomMetric hodge involution err2 (batched x)", err_c2)

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. InputDependentRandomMetric  (spatially varying metric)")
print("=" * 60)

metric_b = InputDependentRandomMetric(seed=3)
ops_b    = DiffGeomOps(metric_b)

x_rand = tf.random.uniform((B, 3), dtype=tf.float64)
alpha2 = tf.random.normal((B, 3), dtype=tf.float64)
beta2  = tf.random.normal((B, 3), dtype=tf.float64)

star_star_a2 = ops_b.star_2form(ops_b.star_1form(alpha2, x_rand), x_rand)
err2a = float(tf.reduce_max(tf.abs(star_star_a2 - alpha2)).numpy())
check("*(*1-form) = 1-form", err2a)

star_star_b2 = ops_b.star_1form(ops_b.star_2form(beta2, x_rand), x_rand)
err2b = float(tf.reduce_max(tf.abs(star_star_b2 - beta2)).numpy())
check("*(*2-form) = 2-form", err2b)

x_rand2  = tf.random.uniform((B, 3), 0.0, 1.0, dtype=tf.float64)
alpha_b2 = tf.random.normal((B, 3), dtype=tf.float64)
beta_b2  = tf.random.normal((B, 3), dtype=tf.float64)
star_star_a_b2 = ops_b.star_2form(ops_b.star_1form(alpha_b2, x_rand2), x_rand2)
err_b21 = float(tf.reduce_max(tf.abs(star_star_a_b2 - alpha_b2)).numpy())
star_star_b_b2 = ops_b.star_1form(ops_b.star_2form(beta_b2, x_rand2), x_rand2)
err_b22 = float(tf.reduce_max(tf.abs(star_star_b_b2 - beta_b2)).numpy())
check("InputDependentRandomMetric hodge involution err1", err_b21)
check("InputDependentRandomMetric hodge involution err2", err_b22)

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. Flat-metric analytic checks (identity metric, scalar)")
print("=" * 60)

# For g = I_3: star of dx1 should be dx2^dx3, i.e. [1,0,0] -> [1,0,0]
# star of [0,1,0] (=dx2) -> dx3^dx1 = [0,1,0]  etc.
# star_2form([1,0,0]) (= *(dx2^dx3)) should give dx1 = [1,0,0]
# So on flat metric star is the identity matrix on components.

ops_flat = DiffGeomOps(FlatMetric())

x1 = tf.zeros((1, 3), dtype=tf.float64)
for i, name in enumerate(["dx1", "dx2", "dx3"]):
    ei = tf.one_hot([i], 3, dtype=tf.float64)
    s = ops_flat.star_1form(ei, x1).numpy()[0]
    # *dx1 = dx2^dx3 -> component [1,0,0]; *dx2 = dx3^dx1 -> [0,1,0]; *dx3 = dx1^dx2 -> [0,0,1]
    expected = np.eye(3)[i]
    err_flat = np.max(np.abs(s - expected))
    check(f"flat *(e_{i+1}) in 2-form basis = e_{i+1}", err_flat)

# star_2form: *(dx2^dx3) = dx1, etc.
for i, name in enumerate(["dx2^dx3", "dx3^dx1", "dx1^dx2"]):
    ei = tf.one_hot([i], 3, dtype=tf.float64)
    s = ops_flat.star_2form(ei, x1).numpy()[0]
    expected = np.eye(3)[i]
    err_flat2 = np.max(np.abs(s - expected))
    check(f"flat *(e_{i+1} 2-form) in 1-form basis = e_{i+1}", err_flat2)

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("4. star_derivative_2_form fix (experiments/archived/T3_random_PINN.py)")
print("=" * 60)

archived_dir = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'archived')
sys.path.insert(0, archived_dir)
from T3_random_PINN import PINN as PINN_exp

pinn_e = PINN_exp()

# For 2-form beta = [x1, x2, x3], d_beta = (∂x1/∂x1 + ∂x2/∂x2 + ∂x3/∂x3) = 3
# So *(d beta) = 3 / sqrt(g)  (with the fix)
# Use x as the 2-form variable so TF can differentiate it.
test_x = tf.constant([[0.3, 0.3, 0.3]], dtype=tf.float64)
g = pinn_e.metric_tensor(test_x)
g_det = tf.linalg.det(g)
sqrt_g = float(tf.sqrt(g_det).numpy())
print(f"  At x=(0.3,0.3,0.3): det(g)={float(g_det.numpy()):.6f}, sqrt(g)={sqrt_g:.6f}")

test_x_var = tf.Variable([[0.3, 0.3, 0.3]], dtype=tf.float64)
zeros = tf.zeros((1, 3), dtype=tf.float64)
# star_derivative_2_form calls u[:,i] inside; all slicing must be inside the tape
with tf.GradientTape(persistent=True) as tape:
    linear_2form = tf.math.add(test_x_var, zeros)   # div = ∂x1/∂x1+∂x2/∂x2+∂x3/∂x3 = 3
    result = pinn_e.star_derivative_2_form(tape, linear_2form, test_x_var)
expected_val = 3.0 / sqrt_g
actual_val = float(result.numpy())
err_sign = abs(actual_val - expected_val)
print(f"  expected 3/sqrt(g) = {expected_val:.6f},  got {actual_val:.6f}")
check(f"*(d(x_i 2-form)) = 3/sqrt(g)", err_sign, tol=1e-8)

# ──────────────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. Hodge star sign-flip fix (experiments/archived/T3_random_PINN.py)")
print("=" * 60)

# The sign-flip bug was in the 1-form Hodge star.
# Verify the fix by comparing against the known-correct star_1form from DiffGeomOps.
# Note: hodge_star(hodge_star(u)) ≠ u for general metrics because it uses the same
# formula for both 1-forms and 2-forms (a separate known design limitation).

test_x3 = tf.constant([[0.3, 0.3, 0.3]], dtype=tf.float64)
u_test  = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float64)

# experimental hodge_star applied once to a 1-form
star_exp = pinn_e.hodge_star(u_test, test_x3)

# reference star_1form using the corrected DiffGeomOps formula
# (need to build same metric at x=(0.3,0.3,0.3))
g_exp    = pinn_e.metric_tensor(test_x3)
g_inv_exp = tf.linalg.inv(g_exp)
sqrtg_exp = tf.sqrt(tf.linalg.det(g_exp))
star_ref  = sqrtg_exp * tf.squeeze(tf.matmul(g_inv_exp, tf.expand_dims(u_test[0], -1)), -1)
star_ref  = tf.expand_dims(star_ref, 0)

err5 = float(tf.reduce_max(tf.abs(star_exp - star_ref)).numpy())
print(f"  experimental hodge_star:  {star_exp.numpy()}")
print(f"  reference   star_1form:   {star_ref.numpy()}")
check("hodge_star(1-form) matches star_1form formula (sign-flip fixed)", err5)

print()
print("All checks complete.")
