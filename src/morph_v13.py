"""
MORPH v13 - Configurable version for ablation studies.

This is v12 with feature flags. Allows turning off each component to measure
its contribution. Used for the ablation study in Phase 2.

Components (all toggleable):
  - gates: continuous gate relaxation (vs fixed topology)
  - l0_pressure: L0 sparsity on gates
  - diverse_init: random topology initialization
  - fitness_shaping: diversity bonus when fitness is sparse
  - restart: IPOP-CMA-ES restart on stagnation
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4


class MorphV13:
    """Configurable MORPH for ablations."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=30, sigma0=1.5,
                 # Feature flags
                 use_gates=True,  # if False, all gates forced ON (fixed topology)
                 use_l0=True,
                 use_diverse_init=True,
                 use_fitness_shaping=True,
                 use_restart=True,
                 # Hyperparams
                 l0_pressure=0.02, l0_threshold=0.1,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 init_topology_diversity=0.3,
                 stagnation_limit=15, sigma_restart_threshold=1e-4,
                 max_restarts=4, pop_doubling=True,
                 fitness_shaping_weight=0.1,
                 fitness_shaping_threshold=0.5,
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.base_pop_size = pop_size
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.use_gates = use_gates
        self.use_l0 = use_l0
        self.use_diverse_init = use_diverse_init
        self.use_fitness_shaping = use_fitness_shaping
        self.use_restart = use_restart
        self.l0_pressure = l0_pressure if use_l0 else 0.0
        self.l0_threshold = l0_threshold
        self.init_gate_logit_on = init_gate_logit_on
        self.init_gate_logit_off = init_gate_logit_off
        self.init_topology_diversity = init_topology_diversity
        self.stagnation_limit = stagnation_limit
        self.sigma_restart_threshold = sigma_restart_threshold
        self.max_restarts = max_restarts
        self.pop_doubling = pop_doubling
        self.fitness_shaping_weight = fitness_shaping_weight if use_fitness_shaping else 0.0
        self.fitness_shaping_threshold = fitness_shaping_threshold

        self.env_name = env_name
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset
        self.is_continuous = is_continuous

        template = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = template.dim
        self.n_conns = template.n_conns
        self.input_ids = template.input_ids
        self.output_ids = template.output_ids

        if self.use_gates:
            if self.use_diverse_init:
                self.center = self._make_diverse_init()
            else:
                self.center = self._make_minimal_init()
        else:
            # No gates: force all gates ON
            self.center = self._make_no_gates_init()
        self._init_cma_state()

        self.generation = 0
        self.restart_count = 0
        self.stagnation = 0
        self.best_genome = None
        self.best_fitness = -np.inf
        self.elite_archive = []
        self._env = None

    def _make_diverse_init(self):
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        for idx, (a, b) in enumerate(g.candidate_conns):
            if a in g.input_ids and b in g.output_ids:
                g.params[idx] = self.init_gate_logit_on if np.random.rand() < 0.7 else self.init_gate_logit_off
                g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
            else:
                g.params[idx] = self.init_gate_logit_on if np.random.rand() < self.init_topology_diversity else self.init_gate_logit_off
                g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
        return g.params.copy()

    def _make_minimal_init(self):
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        for idx, (a, b) in enumerate(g.candidate_conns):
            if a in g.input_ids and b in g.output_ids:
                g.params[idx] = self.init_gate_logit_on
                g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
            else:
                g.params[idx] = self.init_gate_logit_off
                g.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)
        return g.params.copy()

    def _make_no_gates_init(self):
        """All gates forced ON. Topology is fixed (overcomplete graph)."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        for idx, (a, b) in enumerate(g.candidate_conns):
            g.params[idx] = self.init_gate_logit_on  # all on
            g.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)  # small init for many conns
        return g.params.copy()

    def _init_cma_state(self):
        self.sigma = self.sigma0
        self.C = np.ones(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)

        self.mu = max(1, self.pop_size // 2)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= weights.sum()
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights ** 2)

        self.cc = 4.0 / (self.dim + 4)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        self.chiN = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))

        self.cma_generation = 0

    def _get_env(self):
        if self._env is None:
            self._env = gym.make(self.env_name)
        return self._env

    def _behavior_sig_quick(self, params):
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params
        d = g.to_genome_dict()
        net = FeedForwardNet(d['nodes'], d['connections'], d['input_ids'], d['output_ids'])
        env = self._get_env()
        sig = []
        obs, _ = env.reset(seed=self.seed_offset)
        for step in range(min(30, self.max_steps)):
            logits = net.forward(obs)
            if self.is_continuous:
                a = np.tanh(logits)
                a = np.clip(a, env.action_space.low, env.action_space.high)
            else:
                a = int(np.argmax(logits))
            obs, r, terminated, truncated, _ = env.step(a)
            sig.append(obs[:2])
            if terminated or truncated:
                break
        return np.array(sig).flatten() if sig else np.zeros(2)

    def step(self, fitness_fn):
        # 1. Sample
        samples = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.center + self.sigma * np.sqrt(self.C) * z
            samples.append(x)
        samples = np.array(samples)

        # 2. Evaluate
        fits = np.zeros(self.pop_size)
        for i, x in enumerate(samples):
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = x.copy()
            d = g.to_genome_dict()
            fits[i] = max(fitness_fn(d), 1e-6)

        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        mean_fit = float(np.mean(fits))

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = samples[best_idx].copy()
            self.best_genome = g
            self.stagnation = 0
        else:
            self.stagnation += 1

        # 3. Fitness shaping
        fits_std = np.std(fits)
        fits_mean = np.mean(fits)
        if (self.fitness_shaping_weight > 0 and
            fits_std < self.fitness_shaping_threshold * max(fits_mean, 1e-3)):
            sigs = []
            for x in samples:
                s = self._behavior_sig_quick(x)
                sigs.append(s)
            max_len = max(len(s) for s in sigs) if sigs else 1
            sigs_padded = np.zeros((len(sigs), max_len))
            for i, s in enumerate(sigs):
                sigs_padded[i, :len(s)] = s
            k = min(self.pop_size, 8)
            sample_idx = np.random.choice(self.pop_size, k, replace=False)
            diversity = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                diversity[i] = np.mean(np.linalg.norm(sigs_padded[i] - sigs_padded[sample_idx], axis=1))
            if diversity.max() > 0:
                diversity = diversity / diversity.max()
                fits = fits + self.fitness_shaping_weight * max(fits_mean, 1.0) * diversity

        # 4. Sep-CMA-ES update
        order = np.argsort(-fits)
        sorted_samples = samples[order]
        old_mean = self.center.copy()
        self.center = sum(self.weights[i] * sorted_samples[i] for i in range(self.mu))

        y = (self.center - old_mean) / self.sigma
        z = y / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.cma_generation + 1))) < (1.4 + 2.0 / (self.dim + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        delta = (1 - hsig) * self.cc * (2 - self.cc)
        term1 = self.c1 * (self.pc ** 2)
        term2 = self.cmu * sum(self.weights[i] * (sorted_samples[i] - old_mean) ** 2 for i in range(self.mu)) / self.sigma ** 2
        self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
        self.C = np.maximum(self.C, 1e-20)

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        self.sigma = max(self.sigma, 1e-12)

        # 5. L0 pressure
        if self.l0_pressure > 0 and self.use_gates:
            gate_logits = self.center[:self.n_conns]
            weights_vec = self.center[self.n_conns:2 * self.n_conns]
            active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
            gate_logits[active_small] -= self.l0_pressure
            self.center[:self.n_conns] = gate_logits

        # 6. Restart
        if self.use_restart and (self.sigma < self.sigma_restart_threshold or self.stagnation >= self.stagnation_limit) and self.restart_count < self.max_restarts:
            if self.best_genome is not None:
                self.elite_archive.append((self.best_genome.params.copy(), self.best_fitness))
            self.restart_count += 1
            self.stagnation = 0
            if self.pop_doubling:
                self.pop_size = min(self.pop_size * 2, 200)
            if self.use_diverse_init:
                self.center = self._make_diverse_init()
            else:
                self.center = self._make_minimal_init()
            self._init_cma_state()

        self.cma_generation += 1
        self.generation += 1
        return best_fit, mean_fit

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = self.center
        return g.to_genome_dict()
