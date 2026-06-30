"""
core/networks.py — Neural network architecture for harmonic form learning.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

tf.keras.backend.set_floatx('float64')


class SineActivation(Layer):
    """Sine-cosine feature layer that encodes T^3 periodicity.

    Generates sin(2πnx) and cos(2πnx) for n=1..max_freq per input coordinate.
    Output dimension: 3 * 2 * max_freq.
    """
    def __init__(self, max_freq: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.max_freq = max_freq

    def call(self, inputs):
        features = []
        for n in range(1, self.max_freq + 1):
            features.append(tf.sin(2 * n * np.pi * inputs))
            features.append(tf.cos(2 * n * np.pi * inputs))
        return tf.concat(features, axis=1)

    def get_config(self):
        return {**super().get_config(), 'max_freq': self.max_freq}


def build_network(max_freq: int = 1, width: int = 64) -> tf.keras.Model:
    """Harmonic-form network: Input(3) → SineActivation(max_freq) → width→width→width//2→3."""
    return tf.keras.Sequential([
        tf.keras.layers.Input((3,), dtype=tf.float64),
        SineActivation(max_freq=max_freq),
        tf.keras.layers.Dense(width, activation='tanh', dtype=tf.float64),
        tf.keras.layers.Dense(width, activation='tanh', dtype=tf.float64),
        tf.keras.layers.Dense(width // 2, activation='tanh', dtype=tf.float64),
        tf.keras.layers.Dense(3, dtype=tf.float64),
    ])
