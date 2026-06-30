"""
models/zero_finder.py — Zero-finding for learned harmonic 1-forms on T^3.

Uses scipy differential_evolution with multiple restarts and a coarse grid
pre-scan to minimise ||lambda(x)||^2 over [0,1]^3.
"""
import numpy as np
import tensorflow as tf
from scipy.optimize import differential_evolution, minimize


def find_form_zero(model, tol: float = 1e-3, bounds=None, n_restarts: int = 5):
    """
    Find x in T^3 = [0,1]^3 where ||model(x)||^2 is minimised.

    Parameters
    ----------
    model      : callable  (B,3) -> (B,3)
    tol        : float     Threshold below which min_norm counts as a zero.
    bounds     : list of (lo,hi) triples.  Defaults to [(0,1)]*3.
    n_restarts : int       Number of DE restarts with different seeds.

    Returns
    -------
    found    : bool
    min_norm : float   sqrt of the minimum squared norm found
    best_x   : ndarray shape (3,)
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * 3

    def objective(x_np):
        lam = model(tf.constant(x_np[None], dtype=tf.float64))[0].numpy()
        return float(np.dot(lam, lam))

    best_fun = np.inf
    best_x = None

    # Coarse grid pre-scan: find promising starting point
    grid = np.linspace(0.025, 0.975, 20)
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing='ij')
    grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    grid_vals = model(tf.constant(grid_pts, dtype=tf.float64)).numpy()
    grid_norms_sq = np.sum(grid_vals ** 2, axis=1)
    best_grid_idx = np.argmin(grid_norms_sq)

    # Local refinement from grid minimum
    res_local = minimize(objective, grid_pts[best_grid_idx],
                         method='L-BFGS-B', bounds=bounds)
    if res_local.fun < best_fun:
        best_fun = res_local.fun
        best_x = res_local.x

    # Multi-restart differential evolution
    for seed in range(n_restarts):
        res = differential_evolution(
            objective, bounds=bounds,
            maxiter=1000, tol=1e-9, polish=True, seed=seed)
        if res.fun < best_fun:
            best_fun = res.fun
            best_x = res.x

    min_norm = float(np.sqrt(max(best_fun, 0.0)))
    return (min_norm < tol), min_norm, best_x
