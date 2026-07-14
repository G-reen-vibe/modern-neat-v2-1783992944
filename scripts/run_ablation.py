"""
Ablation study: run v13 with different feature flags to measure contribution.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.morph_v13 import MorphV13
from src.eval_fast import evaluate_genome_fast
from scripts.run_neat import ENV_CONFIGS


def run_ablation(env_name, ablation_name, flags, generations=30, pop_size=30, seed=0, eval_episodes=3, log_dir=None):
    cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 10**6, size=eval_episodes)

    def fitness_fn(genome):
        m, _ = evaluate_genome_fast(genome, env_name, n_episodes=eval_episodes,
                                    max_steps=cfg['max_steps'], stochastic=False,
                                    seed_offset=int(env_seeds[0]))
        return m + cfg.get('reward_shift', 0.0)

    is_cont = cfg.get('continuous', False)
    runner = MorphV13(cfg['inputs'], cfg['outputs'], pop_size=pop_size,
                      n_hidden_max=8,
                      env_name=env_name, max_steps=cfg['max_steps'],
                      n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                      is_continuous=is_cont,
                      **flags)

    log = {'algo': f'morph_v13_{ablation_name}', 'env': env_name, 'seed': seed, 'flags': flags,
           'generations': generations, 'history': [],
           'final_eval': None, 'wall_time': 0.0}

    t0 = time.time()
    for gen in range(generations):
        best, mean = runner.step(fitness_fn)
        log['history'].append({'gen': gen, 'best': float(best), 'mean': float(mean)})
        if gen % 10 == 0 or gen == generations - 1:
            print(f"  [{ablation_name:25s} {env_name:14s} s{seed}] gen {gen:3d}  best={best:8.2f}  t={time.time()-t0:.1f}s")
    log['wall_time'] = time.time() - t0

    best_g = runner.best_genome_dict()
    final_scores = []
    for s in range(10):
        m, _ = evaluate_genome_fast(best_g, env_name, n_episodes=1,
                                    max_steps=cfg['max_steps'], stochastic=False,
                                    seed_offset=10000 + s)
        final_scores.append(m)
    log['final_eval'] = {'mean': float(np.mean(final_scores)),
                         'std': float(np.std(final_scores)),
                         'scores': [float(x) for x in final_scores]}

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fname = f'{ablation_name}_{env_name}_s{seed}.json'
        with open(os.path.join(log_dir, fname), 'w') as f:
            json.dump(log, f, indent=2)

    return log


ABLATIONS = {
    'full': {},  # all features on (default)
    'no_gates': {'use_gates': False},
    'no_l0': {'use_l0': False},
    'no_diverse_init': {'use_diverse_init': False},
    'no_fitness_shaping': {'use_fitness_shaping': False},
    'no_restart': {'use_restart': False},
    'only_gates': {'use_l0': False, 'use_diverse_init': False, 'use_fitness_shaping': False, 'use_restart': False},
    'gates_l0': {'use_diverse_init': False, 'use_fitness_shaping': False, 'use_restart': False},
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Acrobot-v1')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--gens', type=int, default=30)
    parser.add_argument('--pop', type=int, default=30)
    parser.add_argument('--ablations', nargs='+', default=list(ABLATIONS.keys()))
    parser.add_argument('--log_dir', default='results/ablations')
    args = parser.parse_args()

    for abl_name in args.ablations:
        flags = ABLATIONS[abl_name]
        for seed in args.seeds:
            print(f"\n=== Ablation: {abl_name} on {args.env} seed={seed} ===")
            run_ablation(args.env, abl_name, flags, args.gens, args.pop, seed, log_dir=args.log_dir)
