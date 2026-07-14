"""
Visualize the actual network topologies evolved by MORPH.
Saves the best genome from a run and plots it as a graph.
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.morph_v14 import MorphV14
from src.morph_v4 import MorphGenomeV4
from src.eval_fast import evaluate_genome_fast
from scripts.run_neat import ENV_CONFIGS


def visualize_network(genome_dict, title, out_path, figsize=(10, 6)):
    """Draw the network as a graph. Inputs on left, outputs on right, hidden in middle."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')

    nodes = genome_dict['nodes']
    connections = genome_dict['connections']
    input_ids = genome_dict['input_ids']
    output_ids = genome_dict['output_ids']
    hidden_ids = [n['id'] for n in nodes if n['type'] == 'hidden']

    # Positions
    pos = {}
    n_in = len(input_ids)
    for i, nid in enumerate(input_ids):
        pos[nid] = (-0.9, 0.8 - 1.6 * i / max(n_in - 1, 1))
    n_out = len(output_ids)
    for i, nid in enumerate(output_ids):
        pos[nid] = (0.9, 0.8 - 1.6 * i / max(n_out - 1, 1))
    n_hid = len(hidden_ids)
    for i, nid in enumerate(hidden_ids):
        pos[nid] = (0, 0.9 - 1.8 * i / max(n_hid, 1) if n_hid > 1 else 0)

    # Draw connections
    for c in connections:
        if not c.get('enabled', True):
            continue
        a, b = c['in'], c['out']
        if a not in pos or b not in pos:
            continue
        w = c['weight']
        color = 'green' if w > 0 else 'red'
        alpha = min(abs(w) / 3.0, 1.0)
        lw = 0.5 + min(abs(w) / 2.0, 3.0)
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=lw, zorder=1)

    # Draw nodes
    for n in nodes:
        nid = n['id']
        if nid not in pos:
            continue
        x, y = pos[nid]
        if n['type'] == 'in':
            circle = Circle((x, y), 0.06, color='#4CAF50', zorder=2)
        elif n['type'] == 'out':
            circle = Circle((x, y), 0.06, color='#2196F3', zorder=2)
        else:
            circle = Circle((x, y), 0.05, color='#FF9800', zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, str(nid), ha='center', va='center', fontsize=8, fontweight='bold', color='white', zorder=3)

    # Legend
    in_patch = mpatches.Patch(color='#4CAF50', label='Input')
    hid_patch = mpatches.Patch(color='#FF9800', label='Hidden')
    out_patch = mpatches.Patch(color='#2196F3', label='Output')
    pos_line = plt.Line2D([0], [0], color='green', linewidth=2, label='Excitatory (+)')
    neg_line = plt.Line2D([0], [0], color='red', linewidth=2, label='Inhibitory (-)')
    ax.legend(handles=[in_patch, hid_patch, out_patch, pos_line, neg_line], loc='lower right', fontsize=8)

    n_conns = len([c for c in connections if c.get('enabled', True)])
    ax.text(0.5, -0.95, f'Active connections: {n_conns}', ha='center', fontsize=9, transform=ax.transAxes)

    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


def run_and_visualize(env_name, seed=0, generations=30, pop_size=30, out_dir='download/figures/topologies'):
    cfg = ENV_CONFIGS[env_name]
    np.random.seed(seed)
    env_seeds = np.random.randint(0, 10**6, size=3)

    def fitness_fn(genome):
        m, _ = evaluate_genome_fast(genome, env_name, n_episodes=3, max_steps=cfg['max_steps'], stochastic=False, seed_offset=int(env_seeds[0]))
        return m + cfg.get('reward_shift', 0.0)

    runner = MorphV14(cfg['inputs'], cfg['outputs'], pop_size=pop_size, n_hidden_max=8,
                      env_name=env_name, max_steps=cfg['max_steps'], n_episodes=3,
                      seed_offset=int(env_seeds[0]), is_continuous=cfg.get('continuous', False))

    os.makedirs(out_dir, exist_ok=True)

    # Save initial topology
    init_genome = runner.best_genome_dict()
    if init_genome:
        visualize_network(init_genome, f'{env_name} - Initial Topology', f'{out_dir}/{env_name}_initial.png')

    # Run and save snapshots
    snapshots = [5, 15, 29]
    for gen in range(generations):
        best, mean = runner.step(fitness_fn)
        if gen in snapshots:
            best_g = runner.best_genome_dict()
            if best_g:
                visualize_network(best_g, f'{env_name} - Generation {gen+1} (best={best:.1f})',
                                  f'{out_dir}/{env_name}_gen{gen+1}.png')

    # Final
    best_g = runner.best_genome_dict()
    if best_g:
        # Get final eval
        finals = []
        for s in range(10):
            m, _ = evaluate_genome_fast(best_g, env_name, n_episodes=1, max_steps=cfg['max_steps'], stochastic=False, seed_offset=10000+s)
            finals.append(m)
        final_mean = np.mean(finals)
        visualize_network(best_g, f'{env_name} - Final Topology (eval={final_mean:.1f})',
                          f'{out_dir}/{env_name}_final.png')

    print(f"\n{env_name}: final eval = {final_mean:.1f}")


if __name__ == '__main__':
    for env in ['CartPole-v1', 'Acrobot-v1']:
        run_and_visualize(env, seed=0, generations=30)
