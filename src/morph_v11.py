"""
MORPH v11 - Clean unified version: IPOP-CMA-ES + gates + L0 + restart diversity.

Round 11: Step back and create a clean, principled version that combines the
best insights from v1-v10:

1. Single CMA-ES center (simpler than multi-species, more elegant)
2. Continuous gate relaxation (the core MORPH idea)
3. L0 sparsity pressure (use-it-or-lose-it)
4. IPOP-CMA-ES restart: when sigma collapses (stagnation), restart with a
   DIFFERENT random topology init and 2x population. This gives the
   exploration diversity of multi-species, but in a cleaner single-center
   framework.
5. Restart memory: keep the best genome from each restart as an "elite
   archive". Final best = max over all restarts.

The key insight: IPOP-CMA-ES is a well-known technique for multi-modal
optimization. Combined with the gate framework and L0 sparsity, it becomes
a clean, principled algorithm for neuroevolution.

This is the "minimal elegant" version of MORPH.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4
from src.morph_v8 import Species


class MorphV11:
    """IPOP-CMA-ES on (gates, weights, biases) with L0 sparsity + restart diversity."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=30, sigma0=0.7,
                 l0_pressure=0.02, l0_threshold=0.1,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 init_topology_diversity=0.3,
                 stagnation_limit=15, sigma_restart_threshold=1e-4,
                 max_restarts=4, pop_doubling=True,
                 random_init_on_restart=True):
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
        self.init_topology_diversity = init_topology_diversity
        self.stagnation_limit = stagnation_limit
        self.sigma_restart_threshold = sigma_restart_threshold
        self.max_restarts = max_restarts
        self.pop_doubling = pop_doubling
        self.random_init_on_restart = random_init_on_restart

        template = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = template.dim
        self.n_conns = template.n_conns

        # Initial center: standard minimal topology (input->output on)
        self.center = self._make_init_center()
        self._init_cma_state()

        self.generation = 0
        self.restart_count = 0
        self.stagnation = 0
        self.best_genome = None
        self.best_fitness = -np.inf
        self.elite_archive = []  # (params, fitness) tuples from previous restarts

    def _make_init_center(self, diverse=False):
        """Create an initial center. If diverse=True, randomize topology."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        if diverse:
            for idx, (a, b) in enumerate(g.candidate_conns):
                if a in g.input_ids and b in g.output_ids:
                    g.params[idx] = self.init_gate_logit_on if np.random.rand() < 0.7 else self.init_gate_logit_off
                    g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
                else:
                    g.params[idx] = self.init_gate_logit_on if np.random.rand() < self.init_topology_diversity else self.init_gate_logit_off
                    g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
        else:
            for idx, (a, b) in enumerate(g.candidate_conns):
                if a in g.input_ids and b in g.output_ids:
                    g.params[idx] = self.init_gate_logit_on
                    g.params[self.n_conns + idx] = np.random.uniform(-1, 1)
                else:
                    g.params[idx] = self.init_gate_logit_off
                    g.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)
        return g.params.copy()

    def _init_cma_state(self):
        """Initialize CMA-ES state for the current center."""
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

        self.cma_generation = 0  # within this restart

    def step(self, fitness_fn):
        # 1. Sample population
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

        # Track best
        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = samples[best_idx].copy()
            self.best_genome = g
            self.stagnation = 0
        else:
            self.stagnation += 1

        # 3. Sep-CMA-ES update
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

        # 4. L0 sparsity pressure
        if self.l0_pressure > 0:
            gate_logits = self.center[:self.n_conns]
            weights_vec = self.center[self.n_conns:2 * self.n_conns]
            active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
            gate_logits[active_small] -= self.l0_pressure
            self.center[:self.n_conns] = gate_logits

        # 5. Check for restart
        if (self.sigma < self.sigma_restart_threshold or self.stagnation >= self.stagnation_limit) and self.restart_count < self.max_restarts:
            # Save elite
            if self.best_genome is not None:
                self.elite_archive.append((self.best_genome.params.copy(), self.best_fitness))
            # Restart
            self.restart_count += 1
            self.stagnation = 0
            if self.pop_doubling:
                self.pop_size = min(self.pop_size * 2, 200)
            if self.random_init_on_restart:
                self.center = self._make_init_center(diverse=True)
            else:
                self.center = self._make_init_center(diverse=False)
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
