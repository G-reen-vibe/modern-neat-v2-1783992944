"""
MORPH v5 - Sep-CMA-ES on unified (gates, weights, biases) with L0 sparsity pressure.

Round 5 key insight: v4 used antithetic gradient estimation with uniform sigma.
This is suboptimal because gate_logits need different step sizes than weights
(gate changes are discrete-ish, weight changes are continuous).

v5 uses Sep-CMA-ES (separable CMA-ES) which adapts a per-parameter step size.
This lets the algorithm automatically discover that:
  - gate_logits need large steps (to cross 0 and flip topology)
  - weights need small steps (to refine)
  - biases need medium steps

Plus the L0 sparsity pressure from v3/v4: active gates with small |weight|
get pushed toward 0 (off). This creates the "use it or lose it" pressure that
naturally minimizes topology.

The result is a single, principled algorithm: CMA-ES on a unified continuous
representation of (topology, weights), with emergent complexification via
sparsity pressure.

Key novelty: gates are first-class citizens in the CMA-ES optimization, with
their own adapted step sizes. This is fundamentally different from NEAT's
discrete topology mutations.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4  # reuse the genome representation


class MorphV5:
    """Sep-CMA-ES on (gates, weights, biases) with L0 sparsity pressure."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=None, sigma0=0.5,
                 l0_pressure=0.02, l0_threshold=0.1,
                 elitism=1,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 weight_init_scale=1.0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max

        self.center = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = self.center.dim

        # Override init: more controlled
        for idx, (a, b) in enumerate(self.center.candidate_conns):
            if a in self.center.input_ids and b in self.center.output_ids:
                self.center.params[idx] = init_gate_logit_on
                self.center.params[self.center.n_conns + idx] = np.random.uniform(-1, 1) * weight_init_scale
            else:
                self.center.params[idx] = init_gate_logit_off
                self.center.params[self.center.n_conns + idx] = np.random.uniform(-0.5, 0.5) * weight_init_scale

        # Population size from dim
        self.pop_size = pop_size or 4 + int(3 * np.log(self.dim))
        self.pop_size = max(self.pop_size, 8)
        self.sigma = sigma0
        self.elitism = elitism
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold

        # Sep-CMA-ES state
        self.C = np.ones(self.dim)  # diagonal covariance (variance per dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.generation = 0

        # CMA-ES constants (need mu first)
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

    def step(self, fitness_fn):
        # 1. Sample population
        samples = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.center.params + self.sigma * np.sqrt(self.C) * z
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

        # 3. Sep-CMA-ES update
        order = np.argsort(-fits)
        sorted_samples = samples[order]
        old_mean = self.center.params.copy()
        # Recombination (top-mu weighted average)
        self.center.params = sum(self.weights[i] * sorted_samples[i] for i in range(self.mu))

        # Cumulation
        y = (self.center.params - old_mean) / self.sigma
        z = y / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < (1.4 + 2.0 / (self.dim + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        # Update diagonal C
        delta = (1 - hsig) * self.cc * (2 - self.cc)
        term1 = self.c1 * (self.pc ** 2)
        term2 = self.cmu * sum(self.weights[i] * (sorted_samples[i] - old_mean) ** 2 for i in range(self.mu)) / self.sigma ** 2
        self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
        self.C = np.maximum(self.C, 1e-20)

        # Update sigma
        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        self.sigma = max(self.sigma, 1e-12)

        # 4. L0 sparsity pressure on gate_logits of the center
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
