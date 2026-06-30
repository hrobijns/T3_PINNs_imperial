"""
core/metrics.py — Riemannian metric providers for T^3.

All providers expose a .tensor(x) method returning (B,3,3) SPD matrices.
"""
import gc
import random
from typing import Optional

import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')


def make_spd_batch(S: tf.Tensor, min_eig: float = 1e-2) -> tf.Tensor:
    """
    Force a batch of symmetric matrices S (B,3,3) to be SPD by clipping eigenvalues.
    """
    w, v = tf.linalg.eigh(S)                      # (B,3), (B,3,3)
    w = tf.clip_by_value(w, min_eig, np.inf)      # (B,3)
    VWT = tf.matmul(v, v, transpose_b=True)       # (B,3,3)  (just to cache; not strictly needed)
    return tf.matmul(v, tf.matmul(tf.linalg.diag(w), v, transpose_b=True))


class InputDependentRandomMetric:
    """
    Generates smooth, spatially varying SPD metrics on T³.
    Each run randomises amplitudes/phases; within a run g(x) varies smoothly with x.
    """

    def __init__(self, seed: Optional[int] = None):
        # tf.Variables so that reseed() via .assign() never triggers tf.function retracing
        self.a = tf.Variable(np.zeros(6), dtype=tf.float64, trainable=False, name='metric_a')
        self.p = tf.Variable(np.zeros(6), dtype=tf.float64, trainable=False, name='metric_p')
        self.floor = 0.6
        self.reseed(seed)

    def reseed(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
        self.a.assign(np.random.uniform(0.1, 0.4, size=6))
        self.p.assign(np.random.uniform(0, 2*np.pi, size=6))

    def tensor(self, x: tf.Tensor) -> tf.Tensor:
        """
        x: (B,3)
        returns g(x): (B,3,3), symmetric positive definite metric
        """
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        B = tf.shape(x)[0]  # batch size

        # periodic features
        s1 = tf.sin(2*np.pi*x[:, 0:1] + self.p[0]); c1 = tf.cos(2*np.pi*x[:, 0:1] + self.p[1])
        s2 = tf.sin(2*np.pi*x[:, 1:2] + self.p[2]); c2 = tf.cos(2*np.pi*x[:, 1:2] + self.p[3])
        s3 = tf.sin(2*np.pi*x[:, 2:3] + self.p[4]); c3 = tf.cos(2*np.pi*x[:, 2:3] + self.p[5])

        # Smooth symmetric components
        h11 = self.floor + self.a[0]*s1 + 0.05*c2
        h22 = self.floor + self.a[1]*s2 + 0.05*c3
        h33 = self.floor + self.a[2]*s3 + 0.05*c1

        h12 = 0.15*(s1*s2) + 0.05*c3
        h13 = 0.15*(s1*s3) - 0.05*c2
        h23 = 0.15*(s2*s3) + 0.05*c1

        # Correct stacking into (B,3,3)
        g = tf.stack([
            tf.concat([h11, h12, h13], axis=1),
            tf.concat([h12, h22, h23], axis=1),
            tf.concat([h13, h23, h33], axis=1)
        ], axis=-2)   # stack along second-to-last axis so we get (B,3,3)

        # Ensure exact symmetry
        g = 0.5 * (g + tf.transpose(g, perm=[0, 2, 1]))

        # Project to SPD by clamping eigenvalues
        g = make_spd_batch(g, min_eig=1e-2)

        return g


def _make_spd(seed: Optional[int] = None) -> tf.Tensor:
    """Random 3x3 SPD matrix with eigenvalues clipped to [0.01, inf)."""
    if seed is not None:
        np.random.seed(seed); tf.random.set_seed(seed)
    A = tf.random.normal((3, 3), dtype=tf.float64)
    S = 0.5 * (A + tf.transpose(A))
    w, v = tf.linalg.eigh(S)
    w = tf.clip_by_value(w, 1e-2, np.inf)
    return tf.matmul(v, tf.matmul(tf.linalg.diag(w), tf.transpose(v)))


class ConstantRandomMetric:
    """Spatially-constant SPD metric g (same 3x3 matrix at every point)."""
    def __init__(self, seed: Optional[int] = None):
        self.g = _make_spd(seed)

    def tensor(self, x: tf.Tensor) -> tf.Tensor:
        B = tf.shape(x)[0]
        return tf.broadcast_to(self.g, [B, 3, 3])


class FlatMetric:
    """Identity metric g = I_3 everywhere on T^3."""
    def tensor(self, x: tf.Tensor) -> tf.Tensor:
        B = tf.shape(x)[0]
        return tf.broadcast_to(tf.eye(3, dtype=tf.float64), [B, 3, 3])


class FourierMetric:
    """
    Spatially-varying metric via truncated Fourier series on T³.

    Each of the 6 independent symmetric components g_{ij}(x) is a linear
    combination of sin(2π n xₖ) and cos(2π n xₖ) for n=1..max_freq, k=1,2,3.

    Uses tf.Variables for reseed-without-retrace.  SPD enforced via make_spd_batch.
    """

    def __init__(self, max_freq: int = 2, seed: Optional[int] = None,
                 diag_amp: float = 0.15, offdiag_amp: float = 0.08, floor: float = 1.0):
        self.max_freq = max_freq
        self.floor    = floor
        self._diag_amp = diag_amp
        self._offdiag_amp = offdiag_amp
        # n_basis = 2 (sin/cos) × max_freq × 3 coords
        self._n_basis = 2 * max_freq * 3
        # Per-basis amplitude scale: 1/n for frequency n
        self._freq_scale = np.array([
            1.0 / n for n in range(1, max_freq + 1)
            for _ in range(3) for _ in range(2)  # 3 coords × 2 sin/cos
        ])
        # 6 components: (g11, g22, g33, g12, g13, g23) × n_basis coefficients
        self.coeffs = tf.Variable(
            np.zeros((6, self._n_basis)), dtype=tf.float64,
            trainable=False, name='fourier_coeffs')
        self.reseed(seed)

    def reseed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
        raw = np.random.uniform(-1.0, 1.0, size=(6, self._n_basis))
        # Scale by 1/n per frequency and different amplitude for diag vs off-diag
        for i in range(6):
            amp = self._diag_amp if i < 3 else self._offdiag_amp
            raw[i] *= amp * self._freq_scale
        self.coeffs.assign(raw)

    def _basis(self, x: tf.Tensor) -> tf.Tensor:
        """Build Fourier basis matrix.  x: (B,3) → (B, n_basis)."""
        fns = []
        for n in range(1, self.max_freq + 1):
            for k in range(3):
                fns.append(tf.sin(2 * np.pi * n * x[:, k:k+1]))
                fns.append(tf.cos(2 * np.pi * n * x[:, k:k+1]))
        return tf.concat(fns, axis=1)  # (B, n_basis)

    def tensor(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x, dtype=tf.float64)
        B  = tf.cast(tf.shape(x)[0], tf.int32)
        phi = self._basis(x)                          # (B, n_basis)
        # comps[k] = phi @ coeffs[k]  →  (B, 6)
        comps = tf.matmul(phi, tf.transpose(self.coeffs))   # (B, 6)
        h11 = self.floor + comps[:, 0:1]
        h22 = self.floor + comps[:, 1:2]
        h33 = self.floor + comps[:, 2:3]
        h12 = comps[:, 3:4]
        h13 = comps[:, 4:5]
        h23 = comps[:, 5:6]
        g = tf.stack([
            tf.concat([h11, h12, h13], axis=1),
            tf.concat([h12, h22, h23], axis=1),
            tf.concat([h13, h23, h33], axis=1),
        ], axis=-2)                                   # (B,3,3)
        g = 0.5 * (g + tf.transpose(g, perm=[0, 2, 1]))
        return make_spd_batch(g, min_eig=1e-2)

    def metric_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return self.tensor(x)


class ConformalFourierMetric:
    """
    Conformal metric g(x) = exp(f(x)) · I₃  where f is a truncated Fourier series.

    SPD is guaranteed by construction — no eigenvalue clipping needed.
    Amplitudes are kept small so exp(f) stays in a reasonable range.
    """

    def __init__(self, max_freq: int = 2, seed: Optional[int] = None):
        self.max_freq = max_freq
        self._n_basis = 2 * max_freq * 3
        self.coeffs = tf.Variable(
            np.zeros(self._n_basis), dtype=tf.float64,
            trainable=False, name='conformal_coeffs')
        self.reseed(seed)

    def reseed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed)
        # Small amplitudes: exp(f) ≈ e^{±0.45} ≈ [0.64, 1.57]
        self.coeffs.assign(np.random.uniform(-0.15, 0.15, size=self._n_basis))

    def _basis(self, x: tf.Tensor) -> tf.Tensor:
        fns = []
        for n in range(1, self.max_freq + 1):
            for k in range(3):
                fns.append(tf.sin(2 * np.pi * n * x[:, k:k+1]))
                fns.append(tf.cos(2 * np.pi * n * x[:, k:k+1]))
        return tf.concat(fns, axis=1)   # (B, n_basis)

    def tensor(self, x: tf.Tensor) -> tf.Tensor:
        x   = tf.convert_to_tensor(x, dtype=tf.float64)
        phi = self._basis(x)                             # (B, n_basis)
        f   = tf.squeeze(tf.matmul(phi, self.coeffs[:, tf.newaxis]), axis=1)  # (B,)
        exp_f = tf.exp(f)                                # (B,)
        I3  = tf.eye(3, dtype=tf.float64)
        return exp_f[:, tf.newaxis, tf.newaxis] * I3[tf.newaxis, :, :]  # (B,3,3)

    def metric_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return self.tensor(x)
