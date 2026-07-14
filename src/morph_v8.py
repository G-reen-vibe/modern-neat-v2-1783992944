"""
MORPH v8 - Multi-population Sep-CMA-ES (NEAT-speciation analog).

Round 8 key insight: NEAT's speciation is what enables its exploration on
MountainCar. Single-center CMA-ES (v5/v6) can't explore multiple regions
simultaneously.

v8 maintains K centers ("species") in parameter space, each with its own
CMA-ES state (sigma, C, pc, ps). Each generation:
  1. Sample pop_size/K individuals from each center (mixed population)
  2. Evaluate fitness + behavior signature for all
  3. Assign each individual to the nearest center (in behavior space)
  4. Update each center using only individuals assigned to it
  5. Periodically: merge centers that have converged (behaviorally similar)
                  split centers that have too many members

This is the "modern" replacement for NEAT's speciation:
- No compatibility threshold (use behavior distance instead)
- No innovation numbers (parameters are continuous)
- No explicit species representatives (use CMA-ES centers)
- No fitness sharing (each center has its own pop_size budget)

Behavior distance is more meaningful than NEAT's gene-disjoint-based
compatibility, because it directly captures "does this individual behave
differently".
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4


class Species:
    """A single CMA-ES species with its own center, sigma, C, etc."""
    def __init__(self, dim, params_init, sigma0=0.5, pop_size=8):
        self.dim = dim
        self.center = params_init.copy()
        self.sigma = sigma0
        self.pop_size = pop_size
        self.C = np.ones(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_params = params_init.copy()
        self.stagnation = 0  # generations since improvement

        # CMA-ES constants
        self.mu = max(1, pop_size // 2)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= weights.sum()
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights ** 2)
        self.cc = 4.0 / (dim + 4)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c1 = 2.0 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs
        self.chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

    def sample(self, n=None):
        n = n or self.pop_size
        samples = []
        for _ in range(n):
            z = np.random.randn(self.dim)
            x = self.center + self.sigma * np.sqrt(self.C) * z
            samples.append(x)
        return np.array(samples)

    def update(self, samples, fits):
        """Update CMA-ES state given samples and their fitnesses."""
        order = np.argsort(-fits)
        sorted_samples = samples[order]
        old_mean = self.center.copy()
        # Recombination (use top-mu)
        mu = min(self.mu, len(sorted_samples))
        if mu < 1:
            return
        weights = self.weights[:mu]
        weights = weights / weights.sum()  # renormalize
        self.center = sum(weights[i] * sorted_samples[i] for i in range(mu))

        y = (self.center - old_mean) / self.sigma
        z = y / np.sqrt(self.C)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) < (1.4 + 2.0 / (self.dim + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        delta = (1 - hsig) * self.cc * (2 - self.cc)
        term1 = self.c1 * (self.pc ** 2)
        term2 = self.cmu * sum(weights[i] * (sorted_samples[i] - old_mean) ** 2 for i in range(mu)) / self.sigma ** 2
        self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
        self.C = np.maximum(self.C, 1e-20)

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        self.sigma = max(self.sigma, 1e-12)

        # Track best
        best_idx = int(np.argmax(fits))
        if fits[best_idx] > self.best_fitness:
            self.best_fitness = float(fits[best_idx])
            self.best_params = samples[best_idx].copy()
            self.stagnation = 0
        else:
            self.stagnation += 1

        self.generation += 1


class MorphV8:
    """Multi-population Sep-CMA-ES (k species)."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=50, n_species=4, sigma0=0.5,
                 l0_pressure=0.02, l0_threshold=0.1,
                 n_probe_obs=20, probe_seed=42,
                 merge_threshold=0.1, stagnation_limit=15,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.n_species = n_species
        self.pop_size = pop_size
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.merge_threshold = merge_threshold
        self.stagnation_limit = stagnation_limit

        # Probe observations
        rng = np.random.RandomState(probe_seed)
        self.probe_obs = rng.uniform(-1, 1, size=(n_probe_obs, n_inputs)).astype(np.float32)

        # Create template for dim
        template = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = template.dim
        self.n_conns = template.n_conns

        # Initialize K species with different random centers
        self.species = []
        per_species_pop = max(4, pop_size // n_species)
        for k in range(n_species):
            center = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
            # Random init for non-input-output gates and weights
            for idx, (a, b) in enumerate(center.candidate_conns):
                if a in center.input_ids and b in center.output_ids:
                    center.params[idx] = init_gate_logit_on + np.random.uniform(-0.5, 0.5)
                    center.params[self.n_conns + idx] = np.random.uniform(-1, 1)
                else:
                    center.params[idx] = init_gate_logit_off + np.random.uniform(-0.5, 0.5)
                    center.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)
            sp = Species(self.dim, center.params, sigma0=sigma0, pop_size=per_species_pop)
            self.species.append(sp)

        self.generation = 0
        self.best_genome = None
        self.best_fitness = -np.inf

    def _behavior_signature(self, params):
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
        # 1. Sample from all species
        all_samples = []
        sample_species = []  # which species each sample came from
        for k, sp in enumerate(self.species):
            samples = sp.sample()
            for s in samples:
                all_samples.append(s)
                sample_species.append(k)
        all_samples = np.array(all_samples)
        sample_species = np.array(sample_species)

        # 2. Evaluate
        fits = np.zeros(len(all_samples))
        for i, x in enumerate(all_samples):
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = x.copy()
            d = g.to_genome_dict()
            fits[i] = max(fitness_fn(d), 1e-6)

        # Track global best
        best_idx = int(np.argmax(fits))
        if fits[best_idx] > self.best_fitness:
            self.best_fitness = float(fits[best_idx])
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = all_samples[best_idx].copy()
            self.best_genome = g

        # 3. Compute behavior signatures
        sigs = np.array([self._behavior_signature(x) for x in all_samples])

        # 4. Reassign individuals to nearest species (by behavior distance to species center behavior)
        species_sigs = np.array([self._behavior_signature(sp.center) for sp in self.species])
        # Distance matrix: (N_pop, K)
        dists = np.linalg.norm(sigs[:, None, :] - species_sigs[None, :, :], axis=2)
        assigned = np.argmin(dists, axis=1)

        # 5. Update each species with its assigned individuals
        for k, sp in enumerate(self.species):
            mask = assigned == k
            if mask.sum() < 2:
                # Not enough members; just sample more from this species next time
                continue
            sp.update(all_samples[mask], fits[mask])

            # L0 sparsity pressure on this species' center
            if self.l0_pressure > 0:
                gate_logits = sp.center[:self.n_conns]
                weights_vec = sp.center[self.n_conns:2 * self.n_conns]
                active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
                gate_logits[active_small] -= self.l0_pressure
                sp.center[:self.n_conns] = gate_logits

        # 6. Periodically merge / restart species
        if self.generation > 0 and self.generation % 10 == 0:
            self._merge_and_restart()

        best_fit = float(np.max(fits))
        mean_fit = float(np.mean(fits))
        self.generation += 1
        return best_fit, mean_fit

    def _merge_and_restart(self):
        # Merge species that are behaviorally similar
        species_sigs = np.array([self._behavior_signature(sp.center) for sp in self.species])
        n = len(self.species)
        to_merge = []
        merged = set()
        for i in range(n):
            if i in merged:
                continue
            for j in range(i + 1, n):
                if j in merged:
                    continue
                d = np.linalg.norm(species_sigs[i] - species_sigs[j])
                if d < self.merge_threshold:
                    to_merge.append((i, j))
                    merged.add(j)
                    break
        # Apply merges: keep i, drop j
        if to_merge:
            keep = [i for i in range(n) if i not in merged]
            self.species = [self.species[i] for i in keep]

        # Restart stagnating species
        for i, sp in enumerate(self.species):
            if sp.stagnation > self.stagnation_limit:
                # Restart from a random location near the global best
                if self.best_genome is not None:
                    new_center = self.best_genome.params + np.random.randn(self.dim) * 0.5
                else:
                    new_center = np.random.randn(self.dim) * 0.5
                self.species[i] = Species(self.dim, new_center, sigma0=0.5, pop_size=sp.pop_size)

        # If we dropped too many species, add new random ones
        while len(self.species) < self.n_species:
            new_center = np.random.randn(self.dim) * 0.5
            self.species.append(Species(self.dim, new_center, sigma0=0.5, pop_size=max(4, self.pop_size // self.n_species)))

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        return self.species[0].best_params
