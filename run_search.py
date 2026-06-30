"""
run_search.py — CLI entry point for harmonic zero search on T^3.

Usage:
    python run_search.py
    python run_search.py --n-trials 50 --epochs 1000 --output-dir results/test_run
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from experiments.runner import ExperimentRunner
from core.metrics import InputDependentRandomMetric, FourierMetric, ConformalFourierMetric

_METRIC_MAP = {
    'random':    InputDependentRandomMetric,
    'fourier':   FourierMetric,
    'conformal': ConformalFourierMetric,
}


def main():
    parser = argparse.ArgumentParser(
        description='Search for a Riemannian metric on T^3 with harmonic form zeros.')
    parser.add_argument('--n-trials',    type=int,   default=150)
    parser.add_argument('--epochs',      type=int,   default=2000)
    parser.add_argument('--n-col',       type=int,   default=2000,
                        help='Number of collocation points')
    parser.add_argument('--pde-tol',     type=float, default=0.05)
    parser.add_argument('--tol',         type=float, default=1e-3)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--output-dir',  type=str,   default='results')
    parser.add_argument('--print-every', type=int,   default=200)
    parser.add_argument('--metric',      type=str,   default='random',
                        choices=list(_METRIC_MAP.keys()),
                        help='Metric family to search over')
    parser.add_argument('--max-freq',    type=int,   default=2,
                        help='Max Fourier frequency (fourier/conformal metrics only)')
    parser.add_argument('--net-freq',    type=int,   default=None,
                        help='Max frequency in network input features (SineActivation). '
                             'Defaults to --max-freq for fourier/conformal metrics, else 1.')
    parser.add_argument('--net-width',   type=int,   default=64,
                        help='Width of hidden layers in the PINN')
    parser.add_argument('--diag-amp',    type=float, default=None,
                        help='Diagonal coefficient amplitude (FourierMetric only)')
    parser.add_argument('--offdiag-amp', type=float, default=None,
                        help='Off-diagonal coefficient amplitude (FourierMetric only)')
    args = parser.parse_args()

    metric_class  = _METRIC_MAP[args.metric]
    metric_kwargs = {}
    if args.metric in ('fourier', 'conformal'):
        metric_kwargs['max_freq'] = args.max_freq
    if args.metric == 'fourier':
        if args.diag_amp is not None:
            metric_kwargs['diag_amp'] = args.diag_amp
        if args.offdiag_amp is not None:
            metric_kwargs['offdiag_amp'] = args.offdiag_amp

    net_freq = args.net_freq
    if net_freq is None:
        net_freq = args.max_freq if args.metric in ('fourier', 'conformal') else 1

    runner = ExperimentRunner(output_dir=args.output_dir)
    runner.run(
        n_trials       = args.n_trials,
        epochs         = args.epochs,
        n_collocations = args.n_col,
        pde_tol        = args.pde_tol,
        tol            = args.tol,
        lr             = args.lr,
        print_every    = args.print_every,
        metric_class   = metric_class,
        net_max_freq   = net_freq,
        net_width      = args.net_width,
        **metric_kwargs,
    )


if __name__ == '__main__':
    main()
