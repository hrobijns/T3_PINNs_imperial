# T3-PINNs: Computational Search for Harmonic 1-Form Zeros on T³

**Hugo Robijns, Meg Dearden-Hellawell — Imperial College London**
*Preliminary research summary — supervised by Daniel Platt and Daattavya Aggarwal*

---

## Quick Start

**Requirements:** Python 3.10+, install dependencies with:
```bash
pip install -r requirements.txt
```


---

## The Mathematical Question

Does there exist a Riemannian metric **g** on the 3-dimensional torus T³ such that **every** harmonic 1-form vanishes at some point?

On any closed Riemannian manifold, the space of harmonic 1-forms is isomorphic to the first de Rham cohomology group H¹(M). On T³ this is 3-dimensional — there are always exactly three linearly independent harmonic 1-forms (up to scaling). The question is whether a metric can be chosen so that each of these three forms has at least one zero somewhere on T³.

This is related to but distinct from the classical question studied by Calabi (1969) and Volkov (2008), who asked: given a closed 1-form, does there exist a metric making it harmonic? We ask the reverse: given the manifold, does there exist a metric such that *all* harmonic 1-forms simultaneously have zeros?

On the flat torus (g = I), harmonic 1-forms are exactly constant — they cannot vanish. The question is whether a sufficiently curved or anisotropic metric can force all three forms to have zeros.

---

## Approach: Physics-Informed Neural Networks (PINNs)

We use PINNs to numerically learn the harmonic 1-forms associated with a given metric, then search over a large family of metrics for one that produces zeros.

### Why PINNs?

The harmonic equation dλ = 0, δλ = 0 (where δ = −⋆d⋆ is the codifferential and ⋆ is the Hodge star) is a PDE system. PINNs solve PDEs by minimising a residual loss over a set of collocation points — they are mesh-free, differentiable, and can be retrained rapidly across different metrics without rebuilding a solver from scratch.

---

## Pipeline Components

### 1. Riemannian Metric Generator (`core/metrics.py`)

We parameterise metrics via **truncated Fourier series on T³**. Each of the 6 independent symmetric components g_ij(x) is written as:

```
g_ij(x) = floor_ij + Σ_{n=1}^{N} Σ_{k=1}^{3} [a_{ijn k} sin(2πn x_k) + b_{ijnk} cos(2πn x_k)]
```

with coefficients scaled as 1/n to ensure smoothness (higher modes contribute less). Diagonal entries have a positive floor to ensure positive definiteness; off-diagonal amplitudes are kept smaller to ensure SPD structure.

**Why Fourier metrics?** Trigonometric polynomials are dense in the space of smooth functions on T³ (Stone–Weierstrass theorem). As max_freq N increases, the family approximates *any* smooth Riemannian metric arbitrarily well in the C^∞ topology. This gives the search a mathematically rigorous justification — searching over Fourier metrics up to frequency N is searching a dense subspace of all smooth metrics.

**Conditioning:** Naive random Fourier coefficients produce near-singular metrics (condition numbers 85–267) on which PINNs fail to converge. We derived amplitude parameters (`diag_amp=0.15`, `offdiag_amp=0.08`, `floor=1.0`) that keep condition numbers in the range 1.5–4.2 while still producing genuinely non-flat, spatially varying geometry. Convergence is 100% reliable in this regime.

**Other metric families also implemented:**
- `ConformalFourierMetric`: g(x) = exp(f(x))·I where f is a Fourier series. SPD by construction; condition number exactly 1.
- `InputDependentRandomMetric`: ad-hoc smooth SPD metrics; used for initial testing only.

---

### 2. PINN Architecture (`core/networks.py`)

Each harmonic 1-form λ: T³ → ℝ³ is represented by a fully-connected neural network:

```
Input (3) → SineActivation → Dense(64, tanh) → Dense(64, tanh) → Dense(32, tanh) → Output(3)
```

**SineActivation** encodes T³ periodicity by mapping x ↦ [sin(2πn·x), cos(2πn·x)] for n = 1…N. This ensures the network can represent functions of the correct periodicity. We match the network's Fourier frequency N to the metric's max_freq (e.g. N=3 for a max_freq=3 Fourier metric), giving the network the representational capacity to track the metric's spatial variation.

All operations use float64 throughout for numerical stability.

Three independent networks are trained simultaneously, one per harmonic 1-form.

---

### 3. Differential Geometry (`core/differential_geometry.py`)

We implement the full exterior calculus machinery needed to express the harmonic conditions:

- **Hodge star on 1-forms**: (⋆α)_k = √|g| (g⁻¹α)_k
- **Hodge star on 2-forms**: via the full (g⁻¹ ⊗ g⁻¹) contraction with the Levi-Civita tensor
- **Exterior derivative of 1-forms**: dα, a 2-form with components (∂_μ α_ν − ∂_ν α_μ)
- **Exterior derivative of 2-forms**: dβ, a 3-form (scalar on T³)
- **Codifferential**: δλ = −⋆d(⋆λ), using the above operators in composition

**Correctness verification** (`verification/verify_hodge_star.py`):
- ⋆(⋆α) = α on 1-forms ✓
- ⋆(⋆β) = β on 2-forms ✓
- Flat metric analytic check (Hodge star = identity) ✓
- Tested on ConstantRandomMetric, InputDependentRandomMetric, FlatMetric

All operators are implemented as batched TensorFlow operations over (B,3,3) metric tensors, enabling efficient gradient computation through tf.GradientTape.

---

### 4. Training Loss (`models/harmonic_basis.py`)

The three networks are trained jointly with the loss:

```
L = Σᵢ (‖dλᵢ‖² + ‖δλᵢ‖²)          [PDE residual — drives forms to be harmonic]
  + W_norm  · Σᵢ (‖λᵢ‖²_g − 1)²    [norm penalty — prevents trivial λ ≡ 0]
  + W_ortho · Σᵢ<ⱼ ⟨λᵢ, λⱼ⟩²_g    [orthogonality — ensures 3 distinct forms]
```

with W_norm = 100, W_ortho = 500. Norms and inner products are L² norms under the metric g.

Training uses Adam with cosine learning rate decay and gradient clipping (clipnorm=1.0). The training collocation grid (2000 random points on T³) is fixed per run; a separate held-out evaluation grid (10,000 points) is used for all post-training checks.

The `tf.function` training step is compiled once per experiment and reused across metric seeds via `tf.Variable`-based metric parameters — reseeding a metric updates the variable values without triggering retracing.

---

### 5. Convergence Verification (`models/harmonic_basis.py`, `verification/`)

After training, we verify convergence rigorously before trusting any zero-finding results:

**Held-out PDE evaluation:** The PDE residual ‖dλ‖² + ‖δλ‖² is evaluated on a 10,000-point held-out grid (seed 9999, never used in training). Only this held-out residual gates whether zero-finding proceeds — not the training residual, which could be artificially low due to overfitting.

**Triviality detection:** We compute the ratio max_x ‖λ(x)‖ / min_x ‖λ(x)‖ over the held-out grid. If this ratio < 5, the form is flagged as near-constant — a failure mode where the network learns an approximately constant form (which has no zero by the argument for flat metrics). Zeros claimed on trivial forms are rejected.

**Gram matrix:** We compute G_ij = ⟨λᵢ, λⱼ⟩_g over the held-out grid and verify it is close to the identity, confirming the three forms are genuinely linearly independent.

**Flat metric baseline** (`verification/verify_flat_metric.py`): On g = I, harmonic 1-forms must be exactly constant (by Fourier analysis on the flat torus). We verify the trained forms are nearly constant (coefficient of variation < 0.10) and have no zeros (min norm > 0.05). This passes ✓.

---

### 6. Zero Finder (`models/zero_finder.py`)

For each converged harmonic form λᵢ, we find the global minimum of ‖λᵢ(x)‖² over [0,1]³ using:

1. **Coarse grid scan**: evaluate the model on a 20³ = 8,000 point grid to find a promising starting point
2. **Local refinement**: L-BFGS-B from the grid minimum
3. **Multi-restart differential evolution**: 5 independent runs with different random seeds, maxiter=1000

The best result across all runs is taken. If the minimum norm is below tolerance (1e-3), the point is provisionally declared a zero.

**Zero verification**: After finding a candidate zero x*, we evaluate the PDE residual at x* and 50 nearby points (radius 0.01). If the local residual exceeds `pde_tol`, the zero is rejected — it may be an artifact of a poorly-converged region of the network rather than a genuine harmonic zero.

Zero finder correctness is verified with analytic test cases (`verification/verify_zero_finder.py`): all 5/5 tests pass, including interior zeros, boundary zeros, multiple zeros, no-zero cases, and periodicity edge cases.

---

### 7. Search Orchestration (`models/harmonic_basis.py`, `experiments/runner.py`)

The search loop:

1. For each metric seed: reseed metric parameters, reset network weights
2. Train for 2000 epochs
3. Evaluate on held-out grid (triviality check, PDE convergence)
4. If converged: run zero finder on all 3 forms
5. Verify any found zeros against local PDE residual
6. If all 3 forms have zeros: report success

Results are saved incrementally as JSONL (one line per trial) for crash safety, with a full `summary.json` and `config.json` at completion. Each run also produces plots: PDE loss per trial, Gram matrix heatmap, and 3D zero locations if found.

---

## Results

### Mini Production Run (36 Trials)

We searched 36 Fourier metrics (12 each at max_freq = 2, 3, 4) with mild amplitudes:

| Config | Converged | Zeros found | Min norm range |
|--------|-----------|-------------|----------------|
| Fourier, freq=2 | 12/12 | 0 | 0.735–0.931 |
| Fourier, freq=3 | 12/12 | 0 | 0.723–0.890 |
| Fourier, freq=4 | 12/12 | 0 | 0.677–0.878 |

**No metric produced a harmonic 1-form with a zero in any trial.**

The minimum pointwise norm of the harmonic forms sits stubbornly at ~0.7–0.9 across all 36 metrics, with no trend toward zero as metric frequency increases. The lowest observed minimum norm was 0.677 (freq=4, seed=5, form 2) — still comfortably far from zero.

PDE residuals scale mildly with frequency (0.003 → 0.011 → 0.019) as expected: higher-frequency metrics require harmonic forms with larger derivatives, increasing the absolute residual. All residuals remain well within the convergence threshold of 0.05 and do not affect the reliability of the zero-finding.

---

## Repository Structure

```
T3-PINNs/
├── core/
│   ├── metrics.py              # FourierMetric, ConformalFourierMetric
│   ├── networks.py             # PINN architecture with periodic SineActivation
│   └── differential_geometry.py# Hodge star, exterior derivatives, L² inner products
├── models/
│   ├── harmonic_basis.py       # Training loop, loss, search orchestration
│   └── zero_finder.py          # Multi-restart global zero finder
├── experiments/
│   └── runner.py               # Experiment orchestration, plots, JSONL logging
├── verification/
│   ├── verify_flat_metric.py   # Baseline: flat metric produces constant forms
│   ├── verify_hodge_star.py    # Unit tests for differential geometry operators
│   ├── verify_training.py      # Gram matrix and memory stability checks
│   ├── verify_zero_finder.py   # Analytic zero-finding tests (5/5 pass)
│   ├── convergence_diagnostic.py# Network/metric config convergence sweep
│   └── frequency_sweep.py      # Fourier frequency vs convergence mapping
├── results/
│   └── mini_prod_freq{2,3,4}/  # Mini production run results (36 trials)
│       ├── config.json         # Run configuration
│       ├── trials.jsonl        # Per-trial results (crash-safe incremental)
│       └── summary.json        # Full run summary
└── run_search.py               # CLI entry point
```

**Dependencies:** TensorFlow 2.16, NumPy, SciPy, Matplotlib. Python 3.10, float64 throughout.
