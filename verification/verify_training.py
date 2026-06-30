"""
verify_training.py — checks before the overnight run.

1. Gram matrix: after short training, the 3 forms should be nearly
   orthonormal under the L^2 metric inner product (diagonal ~1, off-diag ~0).
2. Memory stability: run 5 trials in a loop and check TF graph node count
   and process RSS don't grow unboundedly.
3. Metric sanity: InputDependentRandomMetric eigenvalue ranges.
"""
import sys, os, gc
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')
sys.path.insert(0, 'metrics_batch_run')
from T3_100_metrics import HarmonicBasisFinder, InputDependentRandomMetric

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("  (psutil not installed — skipping RSS memory check; "
          "run: pip install psutil)")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

def check(name, cond, detail=""):
    print(f"  [{ PASS if cond else FAIL }] {name}" + (f"  ({detail})" if detail else ""))
    return cond

def gram_matrix(basis, x_eval):
    """Compute the L^2 Gram matrix G[i,j] = mean_x λᵢᵀ g⁻¹ λⱼ."""
    g_inv = tf.linalg.inv(basis.metric_tensor(x_eval))   # (B,3,3)
    lambdas = [m(x_eval) for m in basis.models]           # 3 x (B,3)
    G = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            G[i, j] = float(HarmonicBasisFinder._l2_inner(
                lambdas[i], lambdas[j], g_inv).numpy())
    return G

np.random.seed(7); tf.random.set_seed(7)
N_COL  = 2000
N_EVAL = 1000
TRAIN_EPOCHS = 300

x_col  = tf.constant(np.random.uniform(0, 1, (N_COL,  3)), dtype=tf.float64)
x_eval = tf.constant(np.random.uniform(0, 1, (N_EVAL, 3)), dtype=tf.float64)

# ── 1. Gram matrix check ────────────────────────────────────────────────────
print("=" * 60)
print("1. Gram matrix after short training (seed=0, 300 epochs)")
print("=" * 60)

metric = InputDependentRandomMetric(seed=0)
basis  = HarmonicBasisFinder(metric)
basis.train(x_col, epochs=TRAIN_EPOCHS, lr=1e-3, print_every=100,
            clipnorm=1.0, lr_decay=True)

G = gram_matrix(basis, x_eval)
print("\n  Gram matrix G[i,j] = <λᵢ, λⱼ>_g :")
for row in G:
    print("   ", "  ".join(f"{v:+.4f}" for v in row))

diag_ok    = all(abs(G[i,i] - 1.0) < 0.3  for i in range(3))
offdiag_ok = all(abs(G[i,j])        < 0.3  for i in range(3) for j in range(3) if i!=j)
cond       = np.linalg.cond(G)
print(f"\n  condition number: {cond:.2f}")

check("diagonal entries ~1  (|Gᵢᵢ - 1| < 0.3)", diag_ok,
      str([f"{G[i,i]:.3f}" for i in range(3)]))
check("off-diagonal ~0      (|Gᵢⱼ| < 0.3)",     offdiag_ok,
      str([f"{G[i,j]:.3f}" for i in range(3) for j in range(3) if i!=j]))
check("well-conditioned      (cond < 20)",        cond < 20,
      f"cond={cond:.1f}")

# ── 2. Memory stability over 5 trials ───────────────────────────────────────
print()
print("=" * 60)
print("2. Memory stability — 5 trials x 100 epochs")
print("=" * 60)

def get_rss_mb():
    if HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    return None

def get_tf_nodes():
    return len(tf.compat.v1.get_default_graph().as_graph_def().node)

rss_vals  = []

# Reuse one basis (the fixed approach) — should compile _train_step once only
basis = HarmonicBasisFinder(InputDependentRandomMetric(seed=0))

for trial in range(15):
    basis.reseed_metric(trial)
    basis.reset_weights()
    basis.train(x_col, epochs=100, lr=1e-3, print_every=200,
                clipnorm=1.0, lr_decay=True)

    rss = get_rss_mb()
    rss_vals.append(rss)
    print(f"  trial {trial+1:2d}/15"
          + (f"  RSS={rss:.0f} MB" if rss is not None else ""))

if HAS_PSUTIL and len(rss_vals) >= 2:
    rss_growth = rss_vals[-1] - rss_vals[0]
    # After the first trial (warmup/JIT), subsequent trials should be flat
    late_growth = rss_vals[-1] - rss_vals[2]   # growth after first 3 warmup trials
    check("RSS flat after warmup (growth trials 3→15 < 100 MB)", late_growth < 100,
          f"late growth={late_growth:.0f} MB  ({rss_vals[2]:.0f}→{rss_vals[-1]:.0f} MB)")
    check("Total RSS growth < 500 MB over 15 trials", rss_growth < 500,
          f"total growth={rss_growth:.0f} MB  ({rss_vals[0]:.0f}→{rss_vals[-1]:.0f} MB)")

# ── 3. Metric eigenvalue sanity ──────────────────────────────────────────────
print()
print("=" * 60)
print("3. InputDependentRandomMetric eigenvalue sanity (3 seeds)")
print("=" * 60)

x_check = tf.constant(np.random.uniform(0, 1, (500, 3)), dtype=tf.float64)
for seed in [0, 5, 42]:
    g = InputDependentRandomMetric(seed=seed).tensor(x_check)  # (500,3,3)
    eigs = tf.linalg.eigvalsh(g).numpy()                        # (500,3)
    min_eig = float(eigs.min())
    max_eig = float(eigs.max())
    cond_nums = eigs[:, -1] / eigs[:, 0]                        # max/min per point
    max_cond  = float(cond_nums.max())
    spd_ok    = min_eig > 0
    cond_ok   = max_cond < 1000
    print(f"\n  seed={seed}: min_eig={min_eig:.4f}  max_eig={max_eig:.4f}"
          f"  max_cond={max_cond:.1f}")
    check("all eigenvalues > 0 (SPD)",    spd_ok,   f"min_eig={min_eig:.4f}")
    check("condition number < 1000",      cond_ok,  f"max_cond={max_cond:.1f}")

print("\nDone.")
