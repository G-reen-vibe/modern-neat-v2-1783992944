"""
Final comprehensive comparison: all algorithms on all envs.
Saves results in a clean format for the report.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval_fast import evaluate_genome_fast
from scripts.run_neat import ENV_CONFIGS
from scripts.run_experiment import run_experiment


def run_final_comparison():
    algorithms = ['neat', 'cma', 'morph_v5', 'morph_v14', 'latent_morph']
    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    seeds = [0, 1, 2]
    generations = {'CartPole-v1': 40, 'Acrobot-v1': 25, 'MountainCar-v0': 40}
    pop_size = 30

    all_results = {}

    for env in envs:
        for algo in algorithms:
            for seed in seeds:
                key = f"{algo}_{env}_s{seed}"
                if os.path.exists(f"results/final_v2/{key}.json"):
                    print(f"  Skipping {key} (exists)")
                    continue
                print(f"\n=== {algo} on {env} seed={seed} ===")
                log = run_experiment(algo, env, generations[env], pop_size, seed,
                                     eval_episodes=3, log_dir="results/final_v2",
                                     n_hidden=8, n_hidden_max=8)
                all_results[key] = log

    # Summarize
    print("\n\n========= FINAL SUMMARY =========")
    from collections import defaultdict
    summary = defaultdict(list)
    for key, log in all_results.items():
        parts = key.split('_')
        algo = '_'.join(parts[:-2])
        env = parts[-2]
        seed = parts[-1]
        rs = {'MountainCar-v0': 200, 'Acrobot-v1': 500}.get(env, 0)
        finals = [s - rs for s in log['final_eval']['scores']]
        summary[(algo, env)].append({
            'seed': seed,
            'final_mean': np.mean(finals),
            'final_std': np.std(finals),
            'wall': log['wall_time']
        })

    print(f"\n{'algo':18s} {'env':16s} {'n':>3s} {'final_mean':>16s} {'wall_mean':>10s}")
    for (algo, env), v in sorted(summary.items()):
        finals = [x['final_mean'] for x in v]
        walls = [x['wall'] for x in v]
        fm = f"{np.mean(finals):.1f}+/-{np.std(finals):.1f}"
        wm = f"{np.mean(walls):.1f}"
        print(f"{algo:18s} {env:16s} {len(v):3d} {fm:>16s} {wm:>10s}")


if __name__ == '__main__':
    run_final_comparison()
