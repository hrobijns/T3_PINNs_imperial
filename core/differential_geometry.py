"""
core/differential_geometry.py — Hodge star and exterior calculus on T^3.

DiffGeomOps wraps a metric provider and exposes all differential-geometric
operations needed to enforce the harmonic 1-form conditions dλ=0, δλ=0.
"""
import numpy as np
import tensorflow as tf

tf.keras.backend.set_floatx('float64')


class DiffGeomOps:
    """
    Hodge star operators and exterior derivatives for 1-forms on T^3.

    All batch operations: inputs are (B, ...) tensors, x is (B, 3).
    The metric is queried via metric_provider.tensor(x) → (B, 3, 3).
    """

    def __init__(self, metric_provider):
        self.metric_provider = metric_provider

    def metric_tensor(self, x: tf.Tensor) -> tf.Tensor:
        return self.metric_provider.tensor(x)  # (B, 3, 3)

    def star_1form(self, alpha: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Hodge star on a 1-form alpha → 2-form.  (*α)_k = sqrt|g| (g^{-1} α)_k"""
        g     = self.metric_tensor(x)
        g_inv = tf.linalg.inv(g)
        sqrtg = tf.sqrt(tf.linalg.det(g))
        v = tf.squeeze(tf.matmul(g_inv, tf.expand_dims(alpha, -1)), -1)
        return v * sqrtg[:, None]

    def star_2form(self, beta: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Hodge star on a 2-form beta=[b23,b31,b12] → 1-form."""
        g     = self.metric_tensor(x)
        g_inv = tf.linalg.inv(g)
        sqrtg = tf.sqrt(tf.linalg.det(g))

        b23, b31, b12 = tf.unstack(beta, axis=1)
        zero = tf.zeros_like(b12)
        bmat = tf.stack([
            tf.stack([zero,  b12,  -b31], axis=1),
            tf.stack([-b12, zero,   b23], axis=1),
            tf.stack([b31, -b23,   zero], axis=1),
        ], axis=1)

        term = tf.einsum('bip,bjq,bpq->bij', g_inv, g_inv, bmat)
        eps  = tf.constant([
            [[0,0,0],[0,0,1],[0,-1,0]],
            [[0,0,-1],[0,0,0],[1,0,0]],
            [[0,1,0],[-1,0,0],[0,0,0]],
        ], dtype=tf.float64)
        return 0.5 * tf.einsum('bij,kij->bk', term, eps) * sqrtg[:, None]

    def star_3form(self, w: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Hodge star on a 3-form (scalar) w → 0-form.  *(w vol) = w / sqrt|g|"""
        sqrtg = tf.sqrt(tf.linalg.det(self.metric_tensor(x)))
        return w / sqrtg[:, None]

    def grad_scalar(self, tape: tf.GradientTape, f: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Exterior derivative of a 0-form (df) → 1-form."""
        return tf.squeeze(tape.batch_jacobian(f, x), axis=1)

    def exterior_derivative_1_form(self, tape: tf.GradientTape,
                                   alpha: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Exterior derivative of a 1-form → 2-form [d23, d31, d12]."""
        J   = tape.batch_jacobian(alpha, x)  # (B,3,3) J[:,i,μ]=∂_μ α_i
        d23 = J[:, 2, 1] - J[:, 1, 2]
        d31 = J[:, 0, 2] - J[:, 2, 0]
        d12 = J[:, 1, 0] - J[:, 0, 1]
        return tf.stack([d23, d31, d12], axis=1)

    def exterior_derivative_2_form(self, tape: tf.GradientTape,
                                   beta: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        """Exterior derivative of a 2-form → 3-form (scalar)."""
        J = tape.batch_jacobian(beta, x)  # (B,3,3) J[:,k,μ]=∂_μ β_k
        w = J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]
        return tf.expand_dims(w, axis=1)

    @staticmethod
    def l2_inner(a: tf.Tensor, b: tf.Tensor, g_inv: tf.Tensor) -> tf.Tensor:
        """L^2 inner product: mean_x a^T g^{-1} b."""
        ginv_b = tf.matmul(g_inv, tf.expand_dims(b, -1))
        return tf.reduce_mean(tf.reduce_sum(tf.expand_dims(a, -1) * ginv_b, axis=1))
