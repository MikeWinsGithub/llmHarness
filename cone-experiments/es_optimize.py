#!/usr/bin/env python3
"""Evolutionary strategy to optimize parameterized cone strategy.
Uses CMA-ES-style optimization with full episode MSE as the signal.

Usage:
  python3 es_optimize.py --d 8 --pop 20 --episodes 500 --generations 200 --workers 12
"""

import numpy as np
import time
import multiprocessing as mp
import argparse
import json

from run_cones import (
    evaluate_single_trial as evaluate_fwd_trial,
    funnel, warmup,
)
from param_strategy import evaluate_param_strategy, generate_seeds

# Parameter space definition
PARAM_NAMES = ['a', 'b', 'threshold', 'chain_bonus', 'slope',
               'depth_pen', 'jump_pen', 'n_warmup', 'reach_min', 'progress_pow']

# Defaults (≈ fwd-merge)
PARAM_DEFAULTS = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0])

# Search bounds [lo, hi]
PARAM_BOUNDS = np.array([
    [0.3, 3.0],   # a
    [0.0, 2.0],   # b
    [0.0, 3.0],   # threshold
    [0.0, 10.0],  # chain_bonus
    [-1.0, 1.0],  # slope
    [0.0, 3.0],   # depth_pen
    [0.0, 5.0],   # jump_pen
    [1.0, 60.0],  # n_warmup
    [0.0, 5.0],   # reach_min
    [0.5, 3.0],   # progress_pow
])

PARAM_SIGMA0 = np.array([0.3, 0.3, 0.5, 1.0, 0.3, 0.5, 0.5, 10.0, 0.5, 0.3])


def vec_to_params(vec):
    """Convert parameter vector to dict, clamping to bounds."""
    vec = np.clip(vec, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
    return {name: float(val) for name, val in zip(PARAM_NAMES, vec)}


def evaluate_params(param_dict, D, seeds, n_workers):
    """Run strategy with given params, return mean MSE ratio."""
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D
    trial_args = [(W, ol, D, s[0], s[1], param_dict) for s in seeds]
    with mp.Pool(n_workers) as pool:
        results = pool.map(evaluate_param_strategy_worker, trial_args)
    mse_vals = np.array([r[0] for r in results]) / k
    return float(np.mean(mse_vals))


def evaluate_param_strategy_worker(args):
    return evaluate_param_strategy(args)


def simple_es(D, n_episodes, pop_size, n_generations, n_workers, seed=42):
    """Simple (μ, λ) evolution strategy."""
    W = funnel(D); ol = [sum(W[:i]) for i in range(D)]; k = D

    # Generate fixed eval seeds (shared across all evaluations for lower variance)
    eval_seeds = generate_seeds(n_episodes)

    # Also compute fwd-merge baseline
    print("Computing fwd-merge baseline...", end=" ", flush=True)
    fwd_args = [(W, ol, D, s[0], s[1]) for s in eval_seeds]
    with mp.Pool(n_workers) as pool:
        fwd_raw = pool.map(evaluate_fwd_trial, fwd_args)
    fwd_ratio = np.mean(np.array(fwd_raw) / k)
    print(f"ratio={fwd_ratio:.5f}\n")

    # Initialize
    rng = np.random.default_rng(seed)
    mean = PARAM_DEFAULTS.copy()
    sigma = PARAM_SIGMA0.copy() * 0.5  # start with moderate exploration
    n_params = len(mean)
    n_elite = pop_size // 4

    best_ever_ratio = float('inf')
    best_ever_params = None
    history = []

    print(f"ES: pop={pop_size}, elite={n_elite}, episodes={n_episodes}, "
          f"params={n_params}, generations={n_generations}")
    print(f"{'Gen':>4s}  {'best':>8s}  {'mean':>8s}  {'Δfwd':>8s}  {'params'}")
    print("-" * 80)

    for gen in range(n_generations):
        # Sample population
        population = []
        for i in range(pop_size):
            noise = rng.standard_normal(n_params) * sigma
            candidate = mean + noise
            candidate = np.clip(candidate, PARAM_BOUNDS[:, 0], PARAM_BOUNDS[:, 1])
            population.append(candidate)

        # Evaluate each candidate
        ratios = []
        for i, vec in enumerate(population):
            params = vec_to_params(vec)
            ratio = evaluate_params(params, D, eval_seeds, n_workers)
            ratios.append(ratio)

        ratios = np.array(ratios)

        # Sort by ratio (lower is better)
        order = np.argsort(ratios)
        elite_idx = order[:n_elite]

        # Update mean toward elite
        elite_vecs = np.array([population[i] for i in elite_idx])
        new_mean = np.mean(elite_vecs, axis=0)

        # Adaptive sigma: based on elite spread
        elite_std = np.std(elite_vecs, axis=0)
        sigma = np.maximum(elite_std * 1.2, PARAM_SIGMA0 * 0.05)  # floor on sigma

        mean = new_mean
        gen_best = ratios[order[0]]
        gen_mean = np.mean(ratios)

        if gen_best < best_ever_ratio:
            best_ever_ratio = gen_best
            best_ever_params = vec_to_params(population[order[0]])

        history.append({
            'gen': gen, 'best': float(gen_best), 'mean': float(gen_mean),
            'best_params': vec_to_params(population[order[0]]),
        })

        if gen % 5 == 0 or gen < 5:
            bp = vec_to_params(population[order[0]])
            defaults = dict(zip(PARAM_NAMES, PARAM_DEFAULTS))
            short = " ".join(f"{k}={v:.2f}" for k, v in bp.items()
                           if abs(v - defaults[k]) > 0.01)
            print(f"{gen:4d}  {gen_best:8.5f}  {gen_mean:8.5f}  {gen_best-fwd_ratio:+.5f}  {short}",
                  flush=True)

    # Final evaluation with more episodes
    print(f"\n{'='*60}")
    print(f"BEST PARAMS (ratio={best_ever_ratio:.5f}, Δfwd={best_ever_ratio-fwd_ratio:+.5f}):")
    for k, v in best_ever_params.items():
        default = dict(zip(PARAM_NAMES, PARAM_DEFAULTS))[k]
        marker = " ***" if abs(v - default) > 0.01 else ""
        print(f"  {k:15s} = {v:8.4f}  (default {default:.1f}){marker}")

    # Save
    result = {
        'D': D, 'fwd_ratio': fwd_ratio,
        'best_ratio': best_ever_ratio,
        'best_params': best_ever_params,
        'delta': best_ever_ratio - fwd_ratio,
        'history': history,
    }
    fname = f"es_result_d{D}.json"
    with open(fname, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {fname}")

    return best_ever_params, best_ever_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import os
    n_workers = args.workers or os.cpu_count() or 4

    print(f"D={args.d}, workers={n_workers}")
    print("Warming up...", end=" ", flush=True)
    warmup()
    W = funnel(args.d); ol = [sum(W[:i]) for i in range(args.d)]
    evaluate_param_strategy((W, ol, args.d, 42, 42, vec_to_params(PARAM_DEFAULTS)))
    print("done.\n")

    simple_es(args.d, args.episodes, args.pop, args.generations, n_workers, args.seed)
