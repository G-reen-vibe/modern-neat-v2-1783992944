"""
Benchmark runner for neuroevolution experiments.
Standardized setup: CartPole-v1 (4->2), MountainCar-v0 (2->3), Acrobot-v1 (6->3).
Pendulum-v1 (continuous, 3->1).
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
from src.eval import evaluate_genome


ENV_CONFIGS = {
    'CartPole-v1':    {'inputs': 4, 'outputs': 2, 'continuous': False, 'max_steps': 500, 'solved': 475.0},
    'MountainCar-v0': {'inputs': 2, 'outputs': 3, 'continuous': False, 'max_steps': 200, 'solved': -110.0},
    'Acrobot-v1':     {'inputs': 6, 'outputs': 3, 'continuous': False, 'max_steps': 500, 'solved': -100.0},
    'Pendulum-v1':    {'inputs': 3, 'outputs': 1, 'continuous': True,  'max_steps': 200, 'solved': -200.0},
}


def run_neat(env_name, generations=50, pop_size=50, seed=0, eval_episodes=5, log_dir=None):
    cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 10**6, size=eval_episodes)

    def fitness_fn(genome):
        # Use deterministic eval (stochastic=False) for fair comparison
        m, _ = evaluate_genome(genome, env_name, n_episodes=eval_episodes,
                               max_steps=cfg['max_steps'], stochastic=False,
                               seed_offset=int(env_seeds[0]))
        return m

    neat = NEAT(cfg['inputs'], cfg['outputs'], pop_size=pop_size)

    log = {'env': env_name, 'seed': seed, 'pop_size': pop_size,
           'generations': generations, 'history': [],
           'best_genome': None, 'final_eval': None, 'wall_time': 0.0}

    t0 = time.time()
    for gen in range(generations):
        best_fit, mean_fit = neat.step(fitness_fn)
        log['history'].append({
            'gen': gen,
            'best': float(best_fit),
            'mean': float(mean_fit),
            'n_species': len(neat.species),
            'n_nodes_mean': float(np.mean([len(g['nodes']) for g in neat.population])),
            'n_conns_mean': float(np.mean([len([c for c in g['connections'] if c['enabled']]) for g in neat.population])),
        })
        if gen % 5 == 0 or gen == generations - 1:
            print(f"  [NEAT {env_name}] gen {gen:3d}  best={best_fit:8.2f}  mean={mean_fit:8.2f}  species={len(neat.species):3d}  "
                  f"nodes={np.mean([len(g['nodes']) for g in neat.population]):.1f}  t={time.time()-t0:.1f}s")
    log['wall_time'] = time.time() - t0

    # Final eval on fresh seeds
    best_genome = max(neat.population, key=lambda g: g['fitness'])
    log['best_genome'] = best_genome
    final_scores = []
    for s in range(10):
        m, _ = evaluate_genome(best_genome, env_name, n_episodes=1,
                               max_steps=cfg['max_steps'], stochastic=False,
                               seed_offset=10000 + s)
        final_scores.append(m)
    log['final_eval'] = {'mean': float(np.mean(final_scores)),
                         'std': float(np.std(final_scores)),
                         'scores': [float(x) for x in final_scores]}

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        # Save without genome
        save_log = {k: v for k, v in log.items() if k != 'best_genome'}
        with open(os.path.join(log_dir, f'neat_{env_name}_{seed}.json'), 'w') as f:
            json.dump(save_log, f, indent=2)

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--gens', type=int, default=50)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps', type=int, default=5)
    parser.add_argument('--log_dir', default='results/baselines')
    args = parser.parse_args()

    log = run_neat(args.env, args.gens, args.pop, args.seed, args.eps, args.log_dir)
    print(f"\n=== FINAL ===")
    print(f"Best final eval: {log['final_eval']['mean']:.2f} +/- {log['final_eval']['std']:.2f}")
    print(f"Wall time: {log['wall_time']:.1f}s")
