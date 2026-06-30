"""
verify_zero_finder.py — unit tests for HarmonicBasisFinder.find_zero.

Strategy: replace the neural network models with analytic 1-forms whose
zeros are known exactly, then check differential_evolution finds them.

Tests:
  1. Single zero in the interior at (0.3, 0.4, 0.5)
  2. Single zero near the boundary at (0.05, 0.9, 0.1)
  3. Multiple zeros — optimizer should find the global minimum
  4. No zero (constant form) — correctly reports not found
  5. Zero exactly on a face: x1=0 (i.e. T^3 periodicity edge)
"""
import sys, os, numpy as np, tensorflow as tf
tf.keras.backend.set_floatx('float64')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.harmonic_basis import HarmonicBasisFinder
from core.metrics import FlatMetric

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, cond, detail=""):
    print(f"  [{ PASS if cond else FAIL }] {name}" + (f"  ({detail})" if detail else ""))
    return cond

def eval_norm(model, pt):
    """Euclidean norm of the form at a single point."""
    lam = model(tf.constant(pt[None], dtype=tf.float64))[0].numpy()
    return float(np.sqrt(np.dot(lam, lam)))

class MockModel:
    """Wraps an analytic function as a Keras-compatible callable."""
    def __init__(self, fn):
        self._fn = fn
    def __call__(self, x):
        return self._fn(x)

def sin_form(target):
    """1-form: λᵢ(x) = sin(2π(xᵢ − tᵢ))  — vanishes at x=target (and periodically)."""
    t = np.array(target)
    def fn(x):
        return tf.stack([
            tf.sin(2*np.pi*(x[:, 0] - t[0])),
            tf.sin(2*np.pi*(x[:, 1] - t[1])),
            tf.sin(2*np.pi*(x[:, 2] - t[2])),
        ], axis=1)
    return fn

# Shared basis finder (metric irrelevant for find_zero — it only calls model(x))
basis = HarmonicBasisFinder(FlatMetric())
# Use n_restarts=1 for fast unit tests (production uses 5)
ZERO_KWARGS = dict(n_restarts=1)

# ── Test 1: interior zero ────────────────────────────────────────────────────
print("\nTest 1: zero in interior at (0.3, 0.4, 0.5)")
target1 = np.array([0.3, 0.4, 0.5])
basis.models[0] = MockModel(sin_form(target1))
found, mn, pt = basis.find_zero(0, tol=1e-6, **ZERO_KWARGS)
check("zero found",              found,                        f"min_norm={mn:.2e}")
check("form near-zero at point", eval_norm(basis.models[0], pt) < 1e-6,
      f"||λ(found)||={eval_norm(basis.models[0], pt):.2e}  found={np.round(pt,4)}")

# ── Test 2: zero near boundary ───────────────────────────────────────────────
print("\nTest 2: zero near boundary at (0.05, 0.9, 0.1)")
target2 = np.array([0.05, 0.9, 0.1])
basis.models[0] = MockModel(sin_form(target2))
found, mn, pt = basis.find_zero(0, tol=1e-6, **ZERO_KWARGS)
check("zero found",              found,                        f"min_norm={mn:.2e}")
check("form near-zero at point", eval_norm(basis.models[0], pt) < 1e-6,
      f"||λ(found)||={eval_norm(basis.models[0], pt):.2e}  found={np.round(pt,4)}")

# ── Test 3: multiple zeros — finds global minimum ────────────────────────────
print("\nTest 3: multiple zeros (sin form has zeros at t and t+0.5)")
# sin_form has zeros at target AND target+0.5 (mod 1) since sin(2π·0.5)=0
# Both are in [0,1]^3 when target=(0.2, 0.2, 0.2) → zeros at (0.2,0.2,0.2) and (0.7,0.7,0.7)
target3 = np.array([0.2, 0.2, 0.2])
basis.models[0] = MockModel(sin_form(target3))
found, mn, pt = basis.find_zero(0, tol=1e-6, **ZERO_KWARGS)
check("zero found",              found,                        f"min_norm={mn:.2e}")
check("form near-zero at point", eval_norm(basis.models[0], pt) < 1e-6,
      f"||λ(found)||={eval_norm(basis.models[0], pt):.2e}  found={np.round(pt,4)}")

# ── Test 4: no zero — correctly reports not found ────────────────────────────
print("\nTest 4: no zero (constant form [1, 1, 1])")
basis.models[0] = MockModel(lambda x: tf.ones((tf.shape(x)[0], 3), dtype=tf.float64))
found, mn, pt = basis.find_zero(0, tol=1e-3, **ZERO_KWARGS)
check("correctly not found",  not found,  f"min_norm={mn:.4f} (should be ~sqrt(3)≈1.73)")

# ── Test 5: periodicity — zero at x1=0 (≡ x1=1 on T^3) ─────────────────────
print("\nTest 5: zero on T^3 boundary x1=0 (≡ x1=1)")
# sin(2π x1) = 0 at x1=0 and x1=0.5; take target=(0, 0.3, 0.6)
target5 = np.array([0.0, 0.3, 0.6])
basis.models[0] = MockModel(sin_form(target5))
found, mn, pt = basis.find_zero(0, tol=1e-6, **ZERO_KWARGS)
# The zero at x1=0 is on the boundary; differential_evolution includes boundaries
# Also there's a zero at x1=0.5, x2=0.8, x3=0.1 (mod 1)
check("zero found",              found,                        f"min_norm={mn:.2e}")
check("form near-zero at point", eval_norm(basis.models[0], pt) < 1e-6,
      f"||λ(found)||={eval_norm(basis.models[0], pt):.2e}  found={np.round(pt,4)}")

print("\nDone.")
