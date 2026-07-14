"""
Unified baseline runner: NEAT, FixedGA, SepCMAES across multiple benchmarks & seeds.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neat import NEAT
from src.ga_baseline import FixedGA
from src.cma_baseline import SepCMAES
from src.eval import evaluate_genome
from scripts.run_neat import ENV_CONFIGS


def run_baseline(algo, env_name, generations=50, pop_size=50, seed=0, eval_episodes=5, log_dir=None,
                 n_hidden=8, stochastic_eval=False):
    cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 10**6, size=eval_episodes)

    def fitness_fn(genome):
        m, _ = evaluate_genome(genome, env_name, n_episodes=eval_episodes,
                               max_steps=cfg['max_steps'], stochastic=stochastic_eval,
                               seed_offset=int(env_seeds[0]))
        return m

    if algo == 'neat':
        runner = NEAT(cfg['inputs'], cfg['outputs'], pop_size=pop_size)
    elif algo == 'ga':
        runner = FixedGA(cfg['inputs'], cfg['outputs'], n_hidden=n_hidden, pop_size=pop_size)
    elif algo == 'cma':
        runner = SepCMAES(cfg['inputs'], cfg['outputs'], n_hidden=n_hidden, pop_size=pop_size)
    else:
        raise ValueError(algo)

    log = {'algo': algo, 'env': env_name, 'seed': seed, 'pop_size': pop_size,
           'generations': generations, 'history': [],
           'final_eval': None, 'wall_time': 0.0,
           'n_params': None}

    t0 = time.time()
    for gen in range(generations):
        best, mean = runner.step(fitness_fn)
        n_nodes = np.mean([len(g['nodes']) for g in runner.population]) if algo in ('neat', 'ga') else (cfg['inputs'] + n_hidden + cfg['outputs'])
        n_conns = np.mean([len([c for c in g['connections'] if c['enabled']]) for g in runner.population]) if algo in ('neat', 'ga') else len(runner.template['connections'])
        n_species = len(runner.species) if algo == 'neat' else 1
        log['history'].append({
            'gen': gen, 'best': float(best), 'mean': float(mean),
            'n_species': n_species,
            'n_nodes_mean': float(n_nodes), 'n_conns_mean': float(n_conns),
        })
        if gen % 5 == 0 or gen == generations - 1:
            print(f"  [{algo} {env_name} s{seed}] gen {gen:3d}  best={best:8.2f}  mean={mean:8.2f}  t={time.time()-t0:.1f}s")
    log['wall_time'] = time.time() - t0

    # Get best genome
    if algo == 'neat' or algo == 'ga':
        best_genome = max(runner.population, key=lambda g: g['fitness'])
        log['n_params'] = sum(len([c for c in best_genome['connections'] if c['enabled']]) for _ in [0]) + len([n for n in best_genome['nodes'] if n['type'] != 'in'])
    else:
        best_genome = runner.best_genome
        log['n_params'] = runner.dim

    # Final eval on fresh seeds
    final_scores = []
    for s in range(10):
        m, _ = evaluate_genome(best_genome, env_name, n_episodes=1,
                               max_steps=cfg['max_steps'], stochastic=stochastic_eval,
                               seed_offset=10000 + s)
        final_scores.append(m)
    log['final_eval'] = {'mean': float(np.mean(final_scores)),
                         'std': float(np.std(final_scores)),
                         'scores': [float(x) for x in final_scores]}

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f'{algo}_{env_name}_{seed}.json'), 'w') as f:
            json.dump(log, f, indent=2)

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['neat', 'ga', 'cma'], default='neat')
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--gens', type=int, default=50)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=8)
    parser.add_argument('--log_dir', default='results/baselines')
    args = parser.parse_args()

    log = run_baseline(args.algo, args.env, args.gens, args.pop, args.seed, args.eps,
                       args.log_dir, args.n_hidden)
    print(f"\n=== FINAL ===")
    print(f"Best final eval: {log['final_eval']['mean']:.2f} +/- {log['final_eval']['std']:.2f}")
    print(f"Wall time: {log['wall_time']:.1f}s")
