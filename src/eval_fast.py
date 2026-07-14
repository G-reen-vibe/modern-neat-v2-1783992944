"""
Optimized evaluation harness using FastNet.
Drop-in replacement for src/eval.py but faster.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from src.fastnet import FastNet
from src.network import policy_action, continuous_action


_ENV_CACHE = {}


def make_env(env_name):
    if env_name not in _ENV_CACHE:
        _ENV_CACHE[env_name] = gym.make(env_name)
    return _ENV_CACHE[env_name]


def evaluate_genome_fast(genome, env_name, n_episodes=5, max_steps=1000,
                         stochastic=False, seed_offset=0):
    """Faster version: uses FastNet with cached topology."""
    net = FastNet(genome['nodes'], genome['connections'],
                  genome['input_ids'], genome['output_ids'])
    env = make_env(env_name)
    is_continuous = hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high') and env.action_space.shape is not None

    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_ret = 0.0
        for step in range(max_steps):
            if is_continuous:
                a = np.tanh(net.forward(obs))
                a = np.clip(a, env.action_space.low, env.action_space.high)
            else:
                logits = net.forward(obs)
                if stochastic:
                    z = logits - np.max(logits)
                    ez = np.exp(z)
                    p = ez / np.sum(ez)
                    a = int(np.random.choice(len(p), p=p))
                else:
                    a = int(np.argmax(logits))
            obs, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            if terminated or truncated:
                break
        returns.append(ep_ret)
    return float(np.mean(returns)), float(np.std(returns))
