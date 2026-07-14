"""
CMA-ES baseline on a fixed topology. Implements a simple (1+1)-CMA-ES style
or separable CMA-ES (diagonal covariance), which is fast and works well on small problems.
Uses scipy.stats for sampling, but optimized to be fast.

We use sep-CMA-ES (Ros & Hansen 2008): only diagonal covariance, which scales to
large parameter counts and is fast.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.ga_baseline import make_fixed_genome


def genome_to_vector(g):
    """Flatten weights + biases into a vector."""
    # Connection weights first (in order), then node biases (excluding input nodes)
    vec = []
    for c in g['connections']:
        vec.append(c['weight'])
    for n in g['nodes']:
        if n['type'] != 'in':
            vec.append(n['bias'])
    return np.array(vec)


def vector_to_genome(g, vec):
    """Update genome in-place from vector."""
    idx = 0
    for c in g['connections']:
        c['weight'] = vec[idx]; idx += 1
    for n in g['nodes']:
        if n['type'] != 'in':
            n['bias'] = vec[idx]; idx += 1


class SepCMAES:
    """Separable CMA-ES on a fixed-topology genome."""
    def __init__(self, n_inputs, n_outputs, n_hidden=8, pop_size=None,
                 sigma0=0.5, elitism=1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        # Determine pop size from dimension
        self.template = make_fixed_genome(n_inputs, n_outputs, n_hidden)
        self.dim = len(genome_to_vector(self.template))
        self.pop_size = pop_size or 4 + int(3 * np.log(self.dim))
        self.pop_size = max(self.pop_size, 8)
        self.elitism = elitism

        self.mean = np.zeros(self.dim)
        self.sigma = sigma0
        self.C = np.ones(self.dim)  # diagonal covariance
        self.pc = np.zeros(self.dim)  # evolution path
        self.ps = np.zeros(self.dim)  # cumulation for sigma
        self.generation = 0

        # Init weights first (sets mu_eff)
        self._init_weights()
        # CMA-ES constants
        self.cc = 4.0 / (self.dim + 4)
        self.cs = (self.mu_eff + 2) / (self.dim + self.mu_eff + 5)
        self.c1 = 2.0 / ((self.dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.dim + 1)) - 1) + self.cs
        self.chiN = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))
        self.E = self.dim  # expected length of N(0, I)

    def _init_weights(self):
        # Default mu = pop_size / 2
        self.mu = self.pop_size // 2
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= weights.sum()
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights ** 2)

    def ask(self):
        """Sample population."""
        samples = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.dim)
            x = self.mean + self.sigma * np.sqrt(self.C) * z
            samples.append(x)
        return samples

    def tell(self, samples, fitnesses):
        """Update mean, C, sigma given sorted samples (descending fitness)."""
        # Sort by descending fitness
        order = np.argsort(fitnesses)[::-1]
        sorted_samples = np.array([samples[i] for i in order])
        # Recombination
        old_mean = self.mean.copy()
        self.mean = sum(self.weights[i] * sorted_samples[i] for i in range(self.mu))
        # Cumulation
        y = (self.mean - old_mean) / self.sigma
        z = y / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < (1.4 + 2.0 / (self.dim + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y
        # Update C (diagonal only)
        # C = (1 - c1 - cmu) C + c1 pc pc^T + cmu sum w_i y_i y_i^T (diagonal)
        delta = (1 - hsig) * self.cc * (2 - self.cc)
        term1 = self.c1 * (self.pc ** 2)
        term2 = self.cmu * sum(self.weights[i] * (sorted_samples[i] - old_mean) ** 2 for i in range(self.mu)) / self.sigma ** 2
        self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
        # Update sigma
        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        self.sigma = max(self.sigma, 1e-12)
        self.generation += 1

    def step(self, fitness_fn):
        samples = self.ask()
        # Build genomes
        genomes = []
        fitnesses = []
        for x in samples:
            g = copy.deepcopy(self.template)
            vector_to_genome(g, x)
            g['fitness'] = max(fitness_fn(g), 1e-6)
            fitnesses.append(g['fitness'])
            genomes.append(g)
        # Elitism: keep best
        best_idx = int(np.argmax(fitnesses))
        best_fit = fitnesses[best_idx]
        mean_fit = float(np.mean(fitnesses))
        self.tell(samples, fitnesses)
        # Keep best genome for final eval
        self.best_genome = genomes[best_idx]
        return best_fit, mean_fit
