"""Run MORPH algorithm on a benchmark."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.morph_v1 import Morph
from src.eval import evaluate_genome
from scripts.run_neat import ENV_CONFIGS


def run_morph(env_name, generations=50, pop_size=50, seed=0, eval_episodes=5,
              log_dir=None, **morph_kwargs):
    cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 10**6, size=eval_episodes)

    def fitness_fn(genome_dict):
        m, _ = evaluate_genome(genome_dict, env_name, n_episodes=eval_episodes,
                               max_steps=cfg['max_steps'], stochastic=False,
                               seed_offset=int(env_seeds[0]))
        return m

    morph = Morph(cfg['inputs'], cfg['outputs'], pop_size=pop_size, **morph_kwargs)

    log = {'algo': 'morph_v1', 'env': env_name, 'seed': seed, 'pop_size': pop_size,
           'generations': generations, 'history': [],
           'final_eval': None, 'wall_time': 0.0}

    t0 = time.time()
    for gen in range(generations):
        best, mean = morph.step(fitness_fn)
        # Count active connections in best individual
        best_g = max(morph.population, key=lambda g: g.fitness)
        n_active = int(np.sum(best_g.gates > 0.5))
        n_active_mean = float(np.mean([np.sum(g.gates > 0.5) for g in morph.population]))
        log['history'].append({
            'gen': gen, 'best': float(best), 'mean': float(mean),
            'n_active_conns_best': n_active,
            'n_active_conns_mean': n_active_mean,
            'gate_temp': morph.gate_temperature(),
        })
        if gen % 5 == 0 or gen == generations - 1:
            print(f"  [MORPH {env_name} s{seed}] gen {gen:3d}  best={best:8.2f}  mean={mean:8.2f}  "
                  f"act_conns_best={n_active:3d}  act_conns_mean={n_active_mean:.1f}  "
                  f"T_gate={morph.gate_temperature():.2f}  t={time.time()-t0:.1f}s")
    log['wall_time'] = time.time() - t0

    # Final eval
    best_g = max(morph.population, key=lambda g: g.fitness)
    best_dict = best_g.to_genome_dict(gate_threshold=0.5)
    final_scores = []
    for s in range(10):
        m, _ = evaluate_genome(best_dict, env_name, n_episodes=1,
                               max_steps=cfg['max_steps'], stochastic=False,
                               seed_offset=10000 + s)
        final_scores.append(m)
    log['final_eval'] = {'mean': float(np.mean(final_scores)),
                         'std': float(np.std(final_scores)),
                         'scores': [float(x) for x in final_scores]}
    log['n_active_conns_final'] = int(np.sum(best_g.gates > 0.5))

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f'morph_v1_{env_name}_{seed}.json'), 'w') as f:
            json.dump(log, f, indent=2)

    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartPole-v1')
    parser.add_argument('--gens', type=int, default=50)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps', type=int, default=5)
    parser.add_argument('--n_hidden_max', type=int, default=16)
    parser.add_argument('--log_dir', default='results/morph')
    args = parser.parse_args()

    log = run_morph(args.env, args.gens, args.pop, args.seed, args.eps,
                    args.log_dir, n_hidden_max=args.n_hidden_max)
    print(f"\n=== FINAL ===")
    print(f"Best final eval: {log['final_eval']['mean']:.2f} +/- {log['final_eval']['std']:.2f}")
    print(f"Active connections in final: {log['n_active_conns_final']}")
    print(f"Wall time: {log['wall_time']:.1f}s")
