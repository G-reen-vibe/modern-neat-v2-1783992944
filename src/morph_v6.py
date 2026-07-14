"""
MORPH v6 - Sep-CMA-ES + behavioral diversity bonus.

Round 6 key insight: v5 has no exploration mechanism beyond CMA-ES's sigma.
For exploration-heavy envs (MountainCar), this is insufficient.

v6 adds a behavioral diversity bonus: each individual's fitness is augmented by
a term proportional to its mean distance (in behavior space) from a sample of
other individuals in the current population. Behavior signature = action
distribution over a fixed probe set of observations.

This is the Quality-Diversity principle integrated into CMA-ES: selection
pressures both toward high fitness AND toward covering diverse behaviors.

This is NOT a naive combination - it's a single algorithm where the CMA-ES
gradient signal naturally incorporates the diversity term. The diversity
term is computed once per generation and added to all fitnesses, so CMA-ES
sees a single (modified) fitness landscape.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4


class MorphV6:
    """Sep-CMA-ES on (gates, weights, biases) with L0 sparsity + behavioral diversity."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=None, sigma0=0.5,
                 l0_pressure=0.02, l0_threshold=0.1,
                 diversity_weight=0.1, n_probe_obs=20, probe_seed=42,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max

        self.center = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = self.center.dim

        # Override init
        for idx, (a, b) in enumerate(self.center.candidate_conns):
            if a in self.center.input_ids and b in self.center.output_ids:
                self.center.params[idx] = init_gate_logit_on
                self.center.params[self.center.n_conns + idx] = np.random.uniform(-1, 1)
            else:
                self.center.params[idx] = init_gate_logit_off
                self.center.params[self.center.n_conns + idx] = np.random.uniform(-0.5, 0.5)

        self.pop_size = pop_size or 4 + int(3 * np.log(self.dim))
        self.pop_size = max(self.pop_size, 8)
        self.sigma = sigma0
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.diversity_weight = diversity_weight

        # Probe observations for behavioral diversity
        rng = np.random.RandomState(probe_seed)
        self.probe_obs = rng.uniform(-1, 1, size=(n_probe_obs, n_inputs)).astype(np.float32)

        # Sep-CMA-ES state
        self.C = np.ones(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.generation = 0

        self.mu = self.pop_size // 2
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

        self.best_genome = None
        self.best_fitness = -np.inf

    def _behavior_signature(self, params):
        """Compute behavior signature: action distribution over probe obs."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params
        d = g.to_genome_dict()
        net = FeedForwardNet(d['nodes'], d['connections'], d['input_ids'], d['output_ids'])
        sigs = []
        for obs in self.probe_obs:
            logits = net.forward(obs)
            z = logits - np.max(logits)
            ez = np.exp(z)
            p = ez / np.sum(ez)
            sigs.append(p)
        return np.concatenate(sigs)

    def step(self, fitness_fn):
        # 1. Sample population
        samples = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.center.params + self.sigma * np.sqrt(self.C) * z
            samples.append(x)
        samples = np.array(samples)

        # 2. Evaluate raw fitness
        raw_fits = np.zeros(self.pop_size)
        for i, x in enumerate(samples):
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = x.copy()
            d = g.to_genome_dict()
            raw_fits[i] = max(fitness_fn(d), 1e-6)

        # 3. Compute behavior signatures and diversity bonus
        if self.diversity_weight > 0:
            sigs = np.array([self._behavior_signature(x) for x in samples])
            # For each individual, compute mean distance to a sample of others
            k = min(self.pop_size, 10)
            sample_idx = np.random.choice(self.pop_size, k, replace=False)
            diversity = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                d = np.mean(np.linalg.norm(sigs[i] - sigs[sample_idx], axis=1))
                diversity[i] = d
            if diversity.max() > 0:
                diversity = diversity / diversity.max()
            # Scale diversity to be comparable to fitness
            # Use fitness std as scale
            f_scale = max(np.std(raw_fits), 1e-3)
            adjusted_fits = raw_fits + self.diversity_weight * f_scale * diversity
        else:
            adjusted_fits = raw_fits.copy()

        best_idx = int(np.argmax(raw_fits))  # track raw fitness for "best"
        best_fit = float(raw_fits[best_idx])
        mean_fit = float(np.mean(raw_fits))

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = samples[best_idx].copy()
            self.best_genome = g

        # 4. Sep-CMA-ES update using ADJUSTED fitness
        order = np.argsort(-adjusted_fits)
        sorted_samples = samples[order]
        old_mean = self.center.params.copy()
        self.center.params = sum(self.weights[i] * sorted_samples[i] for i in range(self.mu))

        y = (self.center.params - old_mean) / self.sigma
        z = y / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < (1.4 + 2.0 / (self.dim + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        delta = (1 - hsig) * self.cc * (2 - self.cc)
        term1 = self.c1 * (self.pc ** 2)
        term2 = self.cmu * sum(self.weights[i] * (sorted_samples[i] - old_mean) ** 2 for i in range(self.mu)) / self.sigma ** 2
        self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
        self.C = np.maximum(self.C, 1e-20)

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        self.sigma = max(self.sigma, 1e-12)

        # 5. L0 sparsity pressure
        if self.l0_pressure > 0:
            gate_logits = self.center.params[:self.center.n_conns]
            weights_vec = self.center.params[self.center.n_conns:2 * self.center.n_conns]
            active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
            gate_logits[active_small] -= self.l0_pressure
            self.center.params[:self.center.n_conns] = gate_logits

        self.generation += 1
        return best_fit, mean_fit

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        return self.center.to_genome_dict()
