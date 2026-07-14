"""
Master experiment runner.
Runs all algorithms (NEAT, GA, CMA, MORPH-v*) across multiple seeds and envs.
Saves all results to results/experiments/ and produces comparison tables.
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
from src.eval_fast import evaluate_genome_fast
from scripts.run_neat import ENV_CONFIGS


def make_runner(algo_name, cfg, n_hidden=8, env_name=None, eval_episodes=3, env_seeds=None, **kwargs):
    """Return (runner, fitness_fn_wrapper, get_best_genome_fn)."""
    if algo_name == 'neat':
        runner = NEAT(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50))
        def get_best():
            return max(runner.population, key=lambda g: g['fitness'])
    elif algo_name == 'ga':
        runner = FixedGA(cfg['inputs'], cfg['outputs'], n_hidden=n_hidden, pop_size=kwargs.get('pop_size', 50))
        def get_best():
            return max(runner.population, key=lambda g: g['fitness'])
    elif algo_name == 'cma':
        runner = SepCMAES(cfg['inputs'], cfg['outputs'], n_hidden=n_hidden, pop_size=kwargs.get('pop_size', 50))
        def get_best():
            return runner.best_genome
    else:
        # Morph variants - import dynamically
        if algo_name == 'morph_v1':
            from src.morph_v1 import Morph
            runner = Morph(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                           n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                best_g = max(runner.population, key=lambda g: g.fitness)
                return best_g.to_genome_dict(gate_threshold=0.5)
        elif algo_name == 'morph_v2':
            from src.morph_v2 import MorphV2
            runner = MorphV2(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v3':
            from src.morph_v3 import MorphV3
            runner = MorphV3(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v4':
            from src.morph_v4 import MorphV4
            runner = MorphV4(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v5':
            from src.morph_v5 import MorphV5
            runner = MorphV5(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v6':
            from src.morph_v6 import MorphV6
            runner = MorphV6(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v7':
            from src.morph_v7 import MorphV7
            runner = MorphV7(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v8':
            from src.morph_v8 import MorphV8
            runner = MorphV8(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v9':
            from src.morph_v9 import MorphV9
            is_cont = cfg.get('continuous', False)
            runner = MorphV9(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                             n_hidden_max=kwargs.get('n_hidden_max', 8),
                             env_name=env_name, max_steps=cfg['max_steps'],
                             n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                             is_continuous=is_cont)
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v10':
            from src.morph_v10 import MorphV10
            is_cont = cfg.get('continuous', False)
            runner = MorphV10(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                              n_hidden_max=kwargs.get('n_hidden_max', 8),
                              env_name=env_name, max_steps=cfg['max_steps'],
                              n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                              is_continuous=is_cont)
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v11':
            from src.morph_v11 import MorphV11
            runner = MorphV11(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                              n_hidden_max=kwargs.get('n_hidden_max', 8))
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v12':
            from src.morph_v12 import MorphV12
            is_cont = cfg.get('continuous', False)
            runner = MorphV12(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                              n_hidden_max=kwargs.get('n_hidden_max', 8),
                              env_name=env_name, max_steps=cfg['max_steps'],
                              n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                              is_continuous=is_cont)
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v14':
            from src.morph_v14 import MorphV14
            is_cont = cfg.get('continuous', False)
            runner = MorphV14(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                              n_hidden_max=kwargs.get('n_hidden_max', 8),
                              env_name=env_name, max_steps=cfg['max_steps'],
                              n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                              is_continuous=is_cont)
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'morph_v15':
            from src.morph_v15 import MorphV15
            is_cont = cfg.get('continuous', False)
            runner = MorphV15(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                              n_hidden_max=kwargs.get('n_hidden_max', 8),
                              env_name=env_name, max_steps=cfg['max_steps'],
                              n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                              is_continuous=is_cont)
            def get_best():
                return runner.best_genome_dict()
        elif algo_name == 'latent_morph':
            from src.latent_morph import LatentMorph
            is_cont = cfg.get('continuous', False)
            runner = LatentMorph(cfg['inputs'], cfg['outputs'], pop_size=kwargs.get('pop_size', 50),
                                  n_hidden_max=kwargs.get('n_hidden_max', 8),
                                  env_name=env_name, max_steps=cfg['max_steps'],
                                  n_episodes=eval_episodes, seed_offset=int(env_seeds[0]),
                                  is_continuous=is_cont)
            def get_best():
                return runner.best_genome_dict()
        else:
            raise ValueError(algo_name)
    return runner, get_best


def run_experiment(algo_name, env_name, generations=50, pop_size=50, seed=0,
                   eval_episodes=5, log_dir=None, n_hidden=8, n_hidden_max=8,
                   eval_freq=1, verbose=True):
    cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 10**6, size=eval_episodes)

    def fitness_fn(genome):
        m, _ = evaluate_genome_fast(genome, env_name, n_episodes=eval_episodes,
                                    max_steps=cfg['max_steps'], stochastic=False,
                                    seed_offset=int(env_seeds[0]))
        return m + cfg.get('reward_shift', 0.0)

    runner, get_best = make_runner(algo_name, cfg, n_hidden=n_hidden,
                                    pop_size=pop_size, n_hidden_max=n_hidden_max,
                                    env_name=env_name, eval_episodes=eval_episodes,
                                    env_seeds=env_seeds)

    log = {'algo': algo_name, 'env': env_name, 'seed': seed, 'pop_size': pop_size,
           'generations': generations, 'eval_episodes': eval_episodes,
           'n_hidden': n_hidden, 'n_hidden_max': n_hidden_max,
           'history': [], 'final_eval': None, 'wall_time': 0.0}

    t0 = time.time()
    for gen in range(generations):
        best, mean = runner.step(fitness_fn)
        # Track topology size
        if hasattr(runner, 'population'):
            if algo_name == 'neat' or algo_name == 'ga':
                n_nodes = float(np.mean([len(g['nodes']) for g in runner.population]))
                n_conns = float(np.mean([len([c for c in g['connections'] if c['enabled']]) for g in runner.population]))
                n_species = len(runner.species) if algo_name == 'neat' else 1
            elif algo_name.startswith('morph'):
                # v1 has .gates, v2+ has .gate_logits
                if hasattr(runner.population[0], 'gates'):
                    n_active = float(np.mean([float(np.sum(g.gates > 0.5)) for g in runner.population]))
                else:
                    n_active = float(np.mean([float(np.sum(g.gate_logits > 0)) for g in runner.population]))
                n_conns = n_active
                n_nodes = n_hidden_max  # placeholder
                n_species = 1
            else:
                n_conns = 0
                n_nodes = 0
                n_species = 1
        else:
            n_conns = 0
            n_nodes = 0
            n_species = 1
        log['history'].append({
            'gen': gen, 'best': float(best), 'mean': float(mean),
            'n_conns': float(n_conns), 'n_nodes': float(n_nodes),
            'n_species': n_species,
        })
        if verbose and (gen % 5 == 0 or gen == generations - 1):
            print(f"  [{algo_name:8s} {env_name:14s} s{seed}] gen {gen:3d}  best={best:8.2f}  mean={mean:8.2f}  t={time.time()-t0:5.1f}s")
    log['wall_time'] = time.time() - t0

    # Final eval on 10 fresh seeds
    best_g = get_best()
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
        fname = f'{algo_name}_{env_name}_s{seed}.json'
        with open(os.path.join(log_dir, fname), 'w') as f:
            json.dump(log, f, indent=2)

    return log


def summarize_results(logs):
    """Group by (algo, env) across seeds and produce summary."""
    summary = {}
    for log in logs:
        key = (log['algo'], log['env'])
        if key not in summary:
            summary[key] = {'final_means': [], 'wall_times': [], 'gens_to_solve': []}
        summary[key]['final_means'].append(log['final_eval']['mean'])
        summary[key]['wall_times'].append(log['wall_time'])
        # Generations to solve: first gen where best >= solved threshold
        solved = ENV_CONFIGS[log['env']]['solved']
        reward_shift = ENV_CONFIGS[log['env']].get('reward_shift', 0.0)
        solved_shifted = solved + reward_shift
        gens_to_solve = next((h['gen'] + 1 for h in log['history'] if h['best'] >= solved_shifted), log['generations'] + 1)
        summary[key]['gens_to_solve'].append(gens_to_solve)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', nargs='+', default=['neat', 'ga', 'cma', 'morph_v1'])
    parser.add_argument('--envs', nargs='+', default=['CartPole-v1'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--gens', type=int, default=50)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--eps', type=int, default=5)
    parser.add_argument('--n_hidden', type=int, default=8)
    parser.add_argument('--n_hidden_max', type=int, default=8)
    parser.add_argument('--log_dir', default='results/experiments')
    args = parser.parse_args()

    all_logs = []
    for env_name in args.envs:
        for algo in args.algos:
            for seed in args.seeds:
                print(f"\n=== {algo} on {env_name} seed={seed} ===")
                log = run_experiment(algo, env_name, args.gens, args.pop, seed,
                                     args.eps, args.log_dir, args.n_hidden, args.n_hidden_max)
                all_logs.append(log)

    summary = summarize_results(all_logs)
    print("\n\n========= SUMMARY =========")
    print(f"{'algo':12s} {'env':16s} {'final_mean':>14s} {'wall_time_s':>12s} {'gens_to_solve':>14s}")
    for (algo, env), s in summary.items():
        fm = f"{np.mean(s['final_means']):.1f}+/-{np.std(s['final_means']):.1f}"
        wt = f"{np.mean(s['wall_times']):.1f}"
        gs = f"{np.mean(s['gens_to_solve']):.1f}+/-{np.std(s['gens_to_solve']):.1f}"
        print(f"{algo:12s} {env:16s} {fm:>14s} {wt:>12s} {gs:>14s}")
