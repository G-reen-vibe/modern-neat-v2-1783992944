"""
Evaluation harness for RL benchmarks.
Supports discrete and continuous action spaces.
Standardized evaluation: every individual evaluated on N episodes with fixed seeds for fairness.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from src.network import FeedForwardNet, policy_action, continuous_action


def make_env(env_name):
    return gym.make(env_name)


def evaluate_genome(genome, env_name, n_episodes=5, max_steps=1000,
                   stochastic=False, seed_offset=0, render=False):
    """Evaluate a genome (dict with 'nodes','connections','input_ids','output_ids').
    Returns mean episode return over n_episodes.
    """
    net = FeedForwardNet(genome['nodes'], genome['connections'],
                         genome['input_ids'], genome['output_ids'])
    env = make_env(env_name)
    # Detect continuous vs discrete
    is_continuous = hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high') and env.action_space.shape is not None

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        for step in range(max_steps):
            if is_continuous:
                a = continuous_action(net, obs)
                a = np.clip(a, env.action_space.low, env.action_space.high)
            else:
                a, _ = policy_action(net, obs, stochastic=stochastic)
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            if terminated or truncated:
                break
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def evaluate_population(population, env_name, n_episodes=5, max_steps=1000,
                         stochastic=False, seed_offset=0):
    """Evaluate a list of genomes. Returns list of (mean, std) tuples."""
    results = []
    for g in population:
        m, s = evaluate_genome(g, env_name, n_episodes, max_steps, stochastic, seed_offset)
        results.append((m, s))
    return results
