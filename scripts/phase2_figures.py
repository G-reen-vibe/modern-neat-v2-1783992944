"""
Phase 2: Deeper analysis visualizations.
- Algorithm comparison radar chart
- Speed vs quality scatter
- Topology size distribution
- Convergence rate analysis
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


def load_all_results():
    """Load results from all directories."""
    results_dirs = ['results/final_clean', 'results/final_v2', 'results/final_cartpole',
                    'results/final_acrobot', 'results/final_mountaincar',
                    'results/round_13', 'results/round_15', 'results/round_15_mc',
                    'results/round_26']
    summary = {}
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            continue
        for f in sorted(os.listdir(results_dir)):
            if not f.endswith('.json'):
                continue
            try:
                parts = f.replace('.json', '').rsplit('_', 2)
                algo = parts[0]
                env = parts[1]
                seed = int(parts[2].replace('s', ''))
                d = json.load(open(os.path.join(results_dir, f)))
                key = (algo, env, seed)
                if key not in summary:
                    summary[key] = {
                        'final_mean': np.mean(d['final_eval']['scores']),
                        'final_std': np.std(d['final_eval']['scores']),
                        'wall': d.get('wall_time', 0),
                        'history': d.get('history', [])
                    }
            except:
                pass
    return summary


def plot_radar(summary, out_path):
    """Radar chart comparing algorithms across 3 envs."""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), constrained_layout=True)

    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    algos = ['neat', 'cma', 'morph_v14', 'latent_morph']
    labels = ['NEAT', 'Sep-CMA-ES', 'MORPH v14', 'Latent MORPH']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd']

    # Normalize each env to [0, 1]
    env_ranges = {
        'CartPole-v1': (400, 500),      # 400=worst, 500=best
        'Acrobot-v1': (-110, -70),      # -110=worst(threshold), -70=best
        'MountainCar-v0': (-200, -130), # -200=worst, -130=best
    }

    for algo, label, color in zip(algos, labels, colors):
        values = []
        for env in envs:
            finals = [v['final_mean'] for (a, e, s), v in summary.items() if a == algo and e == env]
            if not finals:
                values.append(0)
                continue
            mean_val = np.mean(finals)
            lo, hi = env_ranges[env]
            # Normalize to [0, 1] where 1 is best
            if env in ['Acrobot-v1', 'MountainCar-v0']:
                # Lower is worse, higher is better
                norm = (mean_val - lo) / (hi - lo)
            else:
                # Higher is better
                norm = (mean_val - lo) / (hi - lo)
            values.append(max(0, min(1, norm)))

        # Close the radar
        values_closed = values + [values[0]]
        angles = np.linspace(0, 2 * np.pi, len(envs), endpoint=False).tolist()
        angles_closed = angles + [angles[0]]

        ax.plot(angles_closed, values_closed, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles_closed, values_closed, alpha=0.15, color=color)

    ax.set_xticks(angles)
    ax.set_xticklabels([e.replace('-v1', '').replace('-v0', '') for e in envs], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.set_title('Algorithm Comparison (normalized, higher = better)', fontsize=12, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_speed_vs_quality(summary, out_path):
    """Scatter: wall time vs final eval, per env."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    algos = ['neat', 'cma', 'morph_v14', 'latent_morph', 'morph_v5']
    labels = ['NEAT', 'Sep-CMA-ES', 'MORPH v14', 'Latent MORPH', 'MORPH v5']
    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#2ca02c']
    markers = ['o', 's', '^', 'D', 'v']

    for ax, env in zip(axes, envs):
        for algo, label, color, marker in zip(algos, labels, colors, markers):
            finals = []
            walls = []
            for (a, e, s), v in summary.items():
                if a == algo and e == env:
                    finals.append(v['final_mean'])
                    walls.append(v['wall'])
            if not finals:
                continue
            ax.scatter(np.mean(walls), np.mean(finals), c=color, label=label, marker=marker, s=150, edgecolors='black', linewidth=0.5, zorder=3)
            ax.errorbar(np.mean(walls), np.mean(finals), yerr=np.std(finals), fmt='none', c=color, capsize=3, zorder=2)

        ax.set_xlabel('Wall Time (s)')
        ax.set_ylabel('Final Eval Return')
        env_short = env.replace('-v1', '').replace('-v0', '')
        ax.set_title(f'{env_short}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add solved threshold line
        if env == 'CartPole-v1':
            ax.axhline(y=475, color='gray', linestyle='--', alpha=0.5, label='Solved')
        elif env == 'Acrobot-v1':
            ax.axhline(y=-100, color='gray', linestyle='--', alpha=0.5, label='Solved')
        elif env == 'MountainCar-v0':
            ax.axhline(y=-110, color='gray', linestyle='--', alpha=0.5, label='Solved')

    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Speed vs Quality (higher/lower is better, depending on env)', fontsize=13, fontweight='bold')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_convergence_rate(summary, out_path):
    """Plot generations to reach 90% of final performance."""
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    algos = ['neat', 'cma', 'morph_v14', 'latent_morph']
    labels = ['NEAT', 'Sep-CMA-ES', 'MORPH v14', 'Latent MORPH']
    colors = {'neat': '#1f77b4', 'cma': '#ff7f0e', 'morph_v14': '#d62728', 'latent_morph': '#9467bd'}

    # For each algo+env, compute generations to 90% of best
    data = {}
    for algo in algos:
        for env in envs:
            best_history = None
            for (a, e, s), v in summary.items():
                if a == algo and e == env:
                    h = v['history']
                    if h:
                        if best_history is None or len(h) > len(best_history):
                            best_history = h
            if best_history:
                bests = [h['best'] for h in best_history]
                final_best = max(bests)
                target = 0.9 * final_best
                gen_to_90 = next((i for i, b in enumerate(bests) if b >= target), len(bests))
                data[(algo, env)] = gen_to_90

    x = np.arange(len(envs))
    width = 0.2
    for i, (algo, label) in enumerate(zip(algos, labels)):
        vals = [data.get((algo, env), 0) for env in envs]
        ax.bar(x + i * width, vals, width, label=label, color=colors[algo], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([e.replace('-v1', '').replace('-v0', '') for e in envs])
    ax.set_ylabel('Generations to 90% of Best')
    ax.set_title('Convergence Rate (lower = faster)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == '__main__':
    os.makedirs('download/figures', exist_ok=True)
    summary = load_all_results()
    print(f"Loaded {len(summary)} results")

    plot_radar(summary, 'download/figures/radar_comparison.png')
    plot_speed_vs_quality(summary, 'download/figures/speed_vs_quality.png')
    plot_convergence_rate(summary, 'download/figures/convergence_rate.png')

    print("\nPhase 2 figures saved!")
