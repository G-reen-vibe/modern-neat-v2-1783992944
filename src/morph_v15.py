"""
MORPH v15 - Capacity scheduling: gradually increase allowed active gates.

Round 16 key insight: NEAT's complexification (start minimal, add nodes/connections
over time) is a key strength. v14 starts with diverse init (many gates on) which
is the opposite.

v15 introduces CAPACITY SCHEDULING:
- Start with a target capacity K = n_inputs * n_outputs (just input->output)
- Every few generations, increase K by 1
- After CMA-ES update, if active gates > K, turn off the ones with smallest |weight|
- L0 pressure still applies (prunes useless active gates)

This combines:
- NEAT's "start minimal, complexify" principle (via the schedule)
- MORPH's continuous gate relaxation (via the CMA-ES on gate logits)
- L0 sparsity (via the use-it-or-lose-it pressure)

The schedule is the "directed complexification" - topology grows in a controlled
way, directed by the schedule (not random mutation).

This is a single-principle algorithm: the schedule controls growth, L0 controls
pruning, CMA-ES controls optimization. All three work on the same continuous
gate representation.
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4


class MorphV15:
    """v14 + capacity scheduling for directed complexification."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=30, sigma0=1.5,
                 l0_pressure=0.005, l0_threshold=0.05,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 stagnation_limit=8, sigma_restart_threshold=1e-3,
                 max_restarts=5, pop_doubling=True,
                 fitness_shaping_weight=0.15, fitness_shaping_threshold=0.5,
                 # Capacity scheduling
                 capacity_start=None,  # default: n_inputs * n_outputs
                 capacity_growth_rate=1,  # add 1 gate per K generations
                 capacity_growth_interval=2,  # every 2 generations
                 capacity_max=None,  # default: n_conns
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.base_pop_size = pop_size
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.init_gate_logit_on = init_gate_logit_on
        self.init_gate_logit_off = init_gate_logit_off
        self.stagnation_limit = stagnation_limit
        self.sigma_restart_threshold = sigma_restart_threshold
        self.max_restarts = max_restarts
        self.pop_doubling = pop_doubling
        self.fitness_shaping_weight = fitness_shaping_weight
        self.fitness_shaping_threshold = fitness_shaping_threshold
        self.capacity_growth_rate = capacity_growth_rate
        self.capacity_growth_interval = capacity_growth_interval

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

        # Capacity schedule
        self.capacity_min = capacity_start if capacity_start is not None else (n_inputs * n_outputs)
        self.capacity_max = capacity_max if capacity_max is not None else self.n_conns
        self.current_capacity = self.capacity_min

        # Start with minimal init (input->output on, rest off)
        self.center = self._make_minimal_init()
        self._init_cma_state()

        self.generation = 0
        self.restart_count = 0
        self.stagnation = 0
        self.best_genome = None
        self.best_fitness = -np.inf
        self.elite_archive = []
        self._env = None

    def _make_minimal_init(self):
        """Start minimal: only input->output on."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        for idx, (a, b) in enumerate(g.candidate_conns):
            if a in g.input_ids and b in g.output_ids:
                g.params[idx] = self.init_gate_logit_on
                g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
            else:
                g.params[idx] = self.init_gate_logit_off
                g.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)
        return g.params.copy()

    def _make_diverse_init(self):
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        for idx, (a, b) in enumerate(g.candidate_conns):
            if a in g.input_ids and b in g.output_ids:
                g.params[idx] = self.init_gate_logit_on if np.random.rand() < 0.7 else self.init_gate_logit_off
                g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
            else:
                g.params[idx] = self.init_gate_logit_on if np.random.rand() < 0.3 else self.init_gate_logit_off
                g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
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

    def _enforce_capacity(self):
        """If active gates > current_capacity, turn off the smallest-weight ones."""
        gate_logits = self.center[:self.n_conns]
        weights = self.center[self.n_conns:2 * self.n_conns]
        active_mask = gate_logits > 0
        n_active = int(np.sum(active_mask))
        if n_active > self.current_capacity:
            # Find active gates with smallest |weight|
            active_indices = np.where(active_mask)[0]
            active_weights = np.abs(weights[active_indices])
            # Sort by |weight| ascending
            order = np.argsort(active_weights)
            n_to_turn_off = n_active - self.current_capacity
            off_indices = active_indices[order[:n_to_turn_off]]
            gate_logits[off_indices] = -1.0  # turn off
            self.center[:self.n_conns] = gate_logits

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
        if self.l0_pressure > 0:
            gate_logits = self.center[:self.n_conns]
            weights_vec = self.center[self.n_conns:2 * self.n_conns]
            active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
            gate_logits[active_small] -= self.l0_pressure
            self.center[:self.n_conns] = gate_logits

        # 6. Capacity scheduling: increase capacity over time
        if self.generation > 0 and self.generation % self.capacity_growth_interval == 0:
            self.current_capacity = min(self.current_capacity + self.capacity_growth_rate, self.capacity_max)

        # 7. Enforce capacity (turn off excess gates)
        self._enforce_capacity()

        # 8. Restart
        if (self.sigma < self.sigma_restart_threshold or self.stagnation >= self.stagnation_limit) and self.restart_count < self.max_restarts:
            if self.best_genome is not None:
                self.elite_archive.append((self.best_genome.params.copy(), self.best_fitness))
            self.restart_count += 1
            self.stagnation = 0
            if self.pop_doubling:
                self.pop_size = min(self.pop_size * 2, 200)
            # On restart, use diverse init AND reset capacity to allow new topology
            self.center = self._make_diverse_init()
            self.current_capacity = min(self.capacity_min * 2, self.capacity_max)  # start with a bit more capacity
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
