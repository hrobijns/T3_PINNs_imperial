"""
experiments/runner.py — ExperimentRunner with logging, plots, and JSON summary.

Usage:
    runner = ExperimentRunner(output_dir='results/run_001')
    runner.run(n_trials=150, epochs=2000)
"""
import json
import os
import time
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.harmonic_basis import HarmonicBasisFinder, search_for_metric_with_zeros
from core.metrics import InputDependentRandomMetric
from core.differential_geometry import DiffGeomOps

tf.keras.backend.set_floatx('float64')


class ExperimentRunner:
    """
    Wraps search_for_metric_with_zeros and saves:
      - loss_curve.png      (loss vs epoch for converged trials)
      - pde_losses.png      (per-trial max PDE loss bar chart)
      - gram_matrix.png     (Gram matrix heatmap of best metric found)
      - zeros_3d.png        (3D scatter of zero locations, if found)
      - summary.json        (all settings, per-trial results, timings)
    """

    def __init__(self, output_dir: str = 'results'):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = output_dir if output_dir != 'results' else f'results/run_{ts}'
        os.makedirs(self.output_dir, exist_ok=True)
        self._loss_history: list = []   # list of (seed, epoch_losses)
        self._jsonl_path = os.path.join(self.output_dir, 'trials.jsonl')

    def _append_trial_jsonl(self, log: dict) -> None:
        """Append one trial's log as a JSON line — crash-safe incremental save."""
        with open(self._jsonl_path, 'a') as f:
            f.write(json.dumps(log) + '\n')

    # ── Internal: instrument training to capture loss curves ─────────────────

    def _make_instrumented_basis(self, metric_provider) -> HarmonicBasisFinder:
        basis = HarmonicBasisFinder(metric_provider)
        original_train = basis.train

        def instrumented_train(x_col, epochs=500, lr=1e-3, print_every=100,
                               clipnorm=None, lr_decay=False, _seed=None):
            epoch_losses = []

            # Temporarily patch _train_step to record loss
            original_step = basis._train_step

            losses_this_trial = []

            if basis._opts is None:
                if lr_decay:
                    schedule = tf.keras.optimizers.schedules.CosineDecay(
                        lr, decay_steps=epochs, alpha=1e-5 / lr)
                    basis._opts = [tf.keras.optimizers.Adam(schedule, clipnorm=clipnorm)
                                   for _ in basis.models]
                else:
                    basis._opts = [tf.keras.optimizers.Adam(lr, clipnorm=clipnorm)
                                   for _ in basis.models]
                basis._var_lists = [m.trainable_variables for m in basis.models]
                basis._sizes     = [len(v) for v in basis._var_lists]
                basis._all_vars  = [v for vl in basis._var_lists for v in vl]
            else:
                basis._reset_optimizers()

            for ep in range(epochs):
                L = basis._train_step(x_col)
                losses_this_trial.append(float(L.numpy()))
                if ep % print_every == 0:
                    print(f"    Epoch {ep:4d} | Loss {L.numpy():.4e}")

            if _seed is not None:
                self._loss_history.append((_seed, losses_this_trial))

        basis.train = instrumented_train
        return basis

    # ── Plotting helpers ──────────────────────────────────────────────────────

    def _plot_loss_curves(self, trial_logs):
        converged = [t for t in trial_logs if t.get('converged', False)]
        if not converged:
            return
        n = min(len(converged), 9)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        axes = np.array(axes).flatten() if n > 1 else [axes]
        for idx, log in enumerate(converged[:n]):
            seed = log['seed']
            history = next((h for s, h in self._loss_history if s == seed), None)
            if history:
                axes[idx].semilogy(history)
                axes[idx].set_title(f'seed={seed}')
                axes[idx].set_xlabel('epoch')
                axes[idx].set_ylabel('loss')
        for ax in axes[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'loss_curve.png'), dpi=150)
        plt.close(fig)

    def _plot_pde_losses(self, trial_logs):
        seeds = [t['seed'] for t in trial_logs if 'pde_losses' in t]
        maxes = [max(t['pde_losses']) for t in trial_logs if 'pde_losses' in t]
        if not seeds:
            return
        fig, ax = plt.subplots(figsize=(max(8, len(seeds) * 0.3), 4))
        colors = ['green' if t.get('converged') else 'red'
                  for t in trial_logs if 'pde_losses' in t]
        ax.bar(range(len(seeds)), maxes, color=colors)
        ax.axhline(y=0.05, color='k', linestyle='--', label='pde_tol=0.05')
        ax.set_xlabel('trial (seed)')
        ax.set_ylabel('max per-form PDE loss')
        ax.set_title('Convergence per trial  (green=converged, red=not)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'pde_losses.png'), dpi=150)
        plt.close(fig)

    def _plot_gram_matrix(self, basis, x_eval):
        g_inv   = tf.linalg.inv(basis.metric_tensor(x_eval))
        lambdas = [m(x_eval) for m in basis.models]
        G = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                G[i, j] = float(DiffGeomOps.l2_inner(lambdas[i], lambdas[j], g_inv))
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(G, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{G[i,j]:.3f}', ha='center', va='center', fontsize=10)
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(['λ₁','λ₂','λ₃']); ax.set_yticklabels(['λ₁','λ₂','λ₃'])
        ax.set_title('Gram matrix  ⟨λᵢ, λⱼ⟩_g')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, 'gram_matrix.png'), dpi=150)
        plt.close(fig)

    def _plot_zeros_3d(self, results):
        found = [(i, r[2]) for i, r in enumerate(results) if r[0]]
        if not found:
            return
        fig = plt.figure(figsize=(6, 5))
        ax  = fig.add_subplot(111, projection='3d')
        colors = ['red', 'blue', 'green']
        for i, pt in found:
            ax.scatter(*pt, color=colors[i], s=100, label=f'λ_{i+1}')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
        ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.set_zlabel('x₃')
        ax.set_title('Zero locations on T³')
        ax.legend()
        fig.savefig(os.path.join(self.output_dir, 'zeros_3d.png'), dpi=150)
        plt.close(fig)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self,
            n_trials:       int   = 150,
            epochs:         int   = 2000,
            lr:             float = 1e-3,
            tol:            float = 1e-3,
            pde_tol:        float = 0.05,
            n_collocations: int   = 2000,
            clipnorm:       float = 1.0,
            lr_decay:       bool  = True,
            print_every:    int   = 200,
            metric_class          = None,
            net_max_freq:   int   = 1,
            net_width:      int   = 64,
            **metric_kwargs) -> dict:

        if metric_class is None:
            metric_class = InputDependentRandomMetric

        config = dict(n_trials=n_trials, epochs=epochs, lr=lr, tol=tol,
                      pde_tol=pde_tol, n_collocations=n_collocations,
                      clipnorm=clipnorm, lr_decay=lr_decay,
                      metric=metric_class.__name__, **metric_kwargs)
        print(f"\nExperimentRunner — output: {self.output_dir}")
        print(f"Config: {config}\n")

        # Persist config immediately so it's on disk even if the run crashes
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        t0 = time.time()
        result = search_for_metric_with_zeros(
            n_collocations=n_collocations,
            n_trials=n_trials,
            epochs=epochs,
            lr=lr,
            tol=tol,
            pde_tol=pde_tol,
            print_every=print_every,
            metric_class=metric_class,
            net_max_freq=net_max_freq,
            net_width=net_width,
            on_trial_end=self._append_trial_jsonl,
            **metric_kwargs,
        )
        elapsed = time.time() - t0

        trial_logs = result['trial_logs']

        # ── Plots ──────────────────────────────────────────────────────────
        self._plot_pde_losses(trial_logs)

        # Gram matrix: use the last converged trial's basis (approximate)
        # Rebuild a quick basis for the winning seed if found
        if result['found']:
            np.random.seed(0); tf.random.set_seed(0)
            x_eval = tf.constant(np.random.uniform(0,1,(500,3)), dtype=tf.float64)
            metric  = metric_class(seed=result['seed'], **metric_kwargs)
            basis   = HarmonicBasisFinder(metric)
            x_col   = tf.constant(np.random.uniform(0,1,(n_collocations,3)), dtype=tf.float64)
            basis.train(x_col, epochs=epochs, lr=lr, print_every=epochs+1,
                        clipnorm=clipnorm, lr_decay=lr_decay)
            self._plot_gram_matrix(basis, x_eval)
            self._plot_zeros_3d(result['results'])

        # ── Summary JSON ───────────────────────────────────────────────────
        summary = {
            'config':        config,
            'found':         result['found'],
            'winning_seed':  result['seed'],
            'wall_time_s':   round(elapsed, 1),
            'trial_logs':    trial_logs,
        }
        with open(os.path.join(self.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

        # ── Console summary ────────────────────────────────────────────────
        n_converged = sum(1 for t in trial_logs if t.get('converged'))
        print(f"\n{'='*60}")
        print(f"Run complete in {elapsed/3600:.1f} h")
        print(f"  Trials run:       {len(trial_logs)} / {n_trials}")
        print(f"  Converged:        {n_converged}")
        print(f"  Zeros found:      {'YES — seed=' + str(result['seed']) if result['found'] else 'no'}")
        print(f"  Output saved to:  {self.output_dir}/")
        print(f"{'='*60}")

        return result
