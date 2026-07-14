"""
Create visualizations for the report.
- Training curves per environment
- Final performance bar charts
- Topology evolution (active connections over generations)
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_dir):
    """Load all results from a directory. Returns dict (algo, env, seed) -> log."""
    results = {}
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith('.json'):
            continue
        parts = f.replace('.json', '').rsplit('_', 2)
        algo = parts[0]
        env = parts[1]
        seed = int(parts[2].replace('s', ''))
        d = json.load(open(os.path.join(results_dir, f)))
        results[(algo, env, seed)] = d
    return results


def plot_training_curves(results_dirs, env_name, algorithms, title, out_path, max_gens=None, reward_shift=0):
    """Plot training curves (best fitness per generation) for each algorithm."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    colors = {'neat': '#1f77b4', 'cma': '#ff7f0e', 'morph_v5': '#2ca02c', 'morph_v14': '#d62728',
              'morph_v3': '#9467bd', 'morph_v12': '#8c564b', 'ga': '#7f7f7f'}
    labels = {'neat': 'NEAT', 'cma': 'Sep-CMA-ES', 'morph_v5': 'MORPH v5 (minimal)',
              'morph_v14': 'MORPH v14 (full)', 'morph_v3': 'MORPH v3', 'morph_v12': 'MORPH v12',
              'ga': 'Fixed GA'}

    for algo in algorithms:
        # Collect curves from all seeds
        curves = []
        for results_dir in results_dirs:
            for (a, e, s), d in load_results(results_dir).items():
                if a == algo and e == env_name:
                    bests = [h['best'] - reward_shift for h in d['history']]
                    if max_gens:
                        bests = bests[:max_gens]
                    curves.append(bests)
        if not curves:
            continue
        # Pad to same length
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        gens = np.arange(max_len)
        color = colors.get(algo, '#333333')
        label = labels.get(algo, algo)
        ax.plot(gens, mean, color=color, label=label, linewidth=2)
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Best Fitness (return)')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_final_performance_bar(results_dirs, env_name, algorithms, title, out_path, reward_shift=0, threshold=None, higher_better=True):
    """Bar chart of final eval mean+/-std for each algorithm."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    colors = {'neat': '#1f77b4', 'cma': '#ff7f0e', 'morph_v5': '#2ca02c', 'morph_v14': '#d62728',
              'morph_v3': '#9467bd', 'morph_v12': '#8c564b', 'ga': '#7f7f7f'}
    labels = {'neat': 'NEAT', 'cma': 'Sep-CMA-ES', 'morph_v5': 'MORPH v5',
              'morph_v14': 'MORPH v14', 'morph_v3': 'MORPH v3', 'morph_v12': 'MORPH v12',
              'ga': 'Fixed GA'}

    algos = []
    means = []
    stds = []
    cols = []
    for algo in algorithms:
        finals = []
        for results_dir in results_dirs:
            for (a, e, s), d in load_results(results_dir).items():
                if a == algo and e == env_name:
                    finals.append(d['final_eval']['mean'] - reward_shift)
        if not finals:
            continue
        algos.append(labels.get(algo, algo))
        means.append(np.mean(finals))
        stds.append(np.std(finals))
        cols.append(colors.get(algo, '#333333'))

    x = np.arange(len(algos))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=cols, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha='right')
    ax.set_ylabel('Final Eval Return')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    if threshold is not None:
        ax.axhline(y=threshold, color='gray', linestyle='--', label=f'Solved threshold ({threshold})')
        ax.legend()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_topology_evolution(results_dirs, env_name, algorithms, title, out_path, max_gens=None):
    """Plot active connections over generations for each algorithm."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    colors = {'neat': '#1f77b4', 'cma': '#ff7f0e', 'morph_v5': '#2ca02c', 'morph_v14': '#d62728',
              'morph_v3': '#9467bd', 'morph_v12': '#8c564b', 'ga': '#7f7f7f'}
    labels = {'neat': 'NEAT', 'cma': 'Sep-CMA-ES', 'morph_v5': 'MORPH v5',
              'morph_v14': 'MORPH v14', 'morph_v3': 'MORPH v3', 'morph_v12': 'MORPH v12',
              'ga': 'Fixed GA'}

    for algo in algorithms:
        curves = []
        for results_dir in results_dirs:
            for (a, e, s), d in load_results(results_dir).items():
                if a == algo and e == env_name:
                    conns = [h.get('n_conns', h.get('n_active_conns_mean', 0)) for h in d['history']]
                    if max_gens:
                        conns = conns[:max_gens]
                    curves.append(conns)
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            padded[i, :len(c)] = c
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        gens = np.arange(max_len)
        color = colors.get(algo, '#333333')
        label = labels.get(algo, algo)
        ax.plot(gens, mean, color=color, label=label, linewidth=2)
        ax.fill_between(gens, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Active Connections')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    os.makedirs('download/figures', exist_ok=True)

    algos = ['neat', 'cma', 'morph_v5', 'morph_v14']
    results_dirs = ['results/final_cartpole', 'results/final_acrobot', 'results/final_mountaincar',
                    'results/round_13', 'results/round_15_mc', 'results/round_15']

    # CartPole
    plot_training_curves(results_dirs, 'CartPole-v1', algos,
                         'CartPole-v1: Training Curves (5 seeds)',
                         'download/figures/cartpole_training.png', max_gens=50)
    plot_final_performance_bar(['results/final_cartpole'], 'CartPole-v1', algos,
                                'CartPole-v1: Final Performance (5 seeds, 50 gens)',
                                'download/figures/cartpole_final.png', threshold=475)

    # Acrobot
    plot_training_curves(results_dirs, 'Acrobot-v1', algos,
                         'Acrobot-v1: Training Curves (3-4 seeds)',
                         'download/figures/acrobot_training.png', max_gens=30, reward_shift=500)
    plot_final_performance_bar(['results/final_acrobot', 'results/round_13'], 'Acrobot-v1', algos,
                                'Acrobot-v1: Final Performance (3-4 seeds)',
                                'download/figures/acrobot_final.png', reward_shift=500, threshold=-100, higher_better=False)

    # MountainCar
    plot_training_curves(results_dirs, 'MountainCar-v0', algos,
                         'MountainCar-v0: Training Curves (3 seeds)',
                         'download/figures/mountaincar_training.png', max_gens=50, reward_shift=200)
    plot_final_performance_bar(['results/final_mountaincar', 'results/round_15_mc'], 'MountainCar-v0', algos,
                                'MountainCar-v0: Final Performance (3 seeds)',
                                'download/figures/mountaincar_final.png', reward_shift=200, threshold=-110, higher_better=False)

    # Topology evolution
    plot_topology_evolution(results_dirs, 'Acrobot-v1', algos,
                            'Acrobot-v1: Topology Evolution (active connections)',
                            'download/figures/acrobot_topology.png', max_gens=30)
    plot_topology_evolution(results_dirs, 'MountainCar-v0', algos,
                            'MountainCar-v0: Topology Evolution (active connections)',
                            'download/figures/mountaincar_topology.png', max_gens=50)

    print("\nAll figures saved to download/figures/")
