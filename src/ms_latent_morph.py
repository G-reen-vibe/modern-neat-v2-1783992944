"""
Multi-Species Latent MORPH - Latent space evolution with multiple species.

Round 29: Combine Latent MORPH (fast, low-dim) with multi-species (exploration).

The key insight: Latent MORPH is fast but single-center, so it struggles on
exploration-heavy tasks (MountainCar). Multi-species CMA-ES in latent space
gives both speed AND exploration.

K species, each with its own latent mean + CMA-ES state. Each generation:
1. Sample from all K species (mixed population)
2. Evaluate fitness + behavior signature
3. Assign each individual to nearest species (by behavior)
4. Update each species with its assigned individuals
5. Merge similar species, restart stagnating ones

The latent space makes the multi-species approach more efficient:
- Lower dim → faster CMA-ES per species
- Full covariance feasible → better optimization
- Behavior distance in latent space is meaningful
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v4 import MorphGenomeV4
from src.morph_v8 import Species
from src.latent_morph import LatentMorph


class MultiSpeciesLatentMorph:
    """Multi-species CMA-ES in latent space."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 latent_dim=128,
                 pop_size=50, n_species=4, sigma0=0.5,
                 decoder_hidden_dim=128, decoder_seed=42,
                 l0_pressure=0.005, l0_threshold=0.05,
                 merge_threshold=0.5, stagnation_limit=10,
                 fitness_shaping_weight=0.15, fitness_shaping_threshold=0.5,
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.latent_dim = latent_dim
        self.n_species = n_species
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.merge_threshold = merge_threshold
        self.stagnation_limit = stagnation_limit
        self.fitness_shaping_weight = fitness_shaping_weight
        self.fitness_shaping_threshold = fitness_shaping_threshold

        self.env_name = env_name
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset
        self.is_continuous = is_continuous

        # Build template
        template = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = template.dim
        self.n_conns = template.n_conns

        # Build fixed decoder (same as LatentMorph)
        rng = np.random.RandomState(decoder_seed)
        self.decoder_W1 = rng.randn(latent_dim, decoder_hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.decoder_b1 = np.zeros(decoder_hidden_dim)
        self.decoder_W2 = rng.randn(decoder_hidden_dim, self.dim) * np.sqrt(2.0 / decoder_hidden_dim)
        self.decoder_b2 = np.zeros(self.dim)
        self.output_scale = 1.0

        # Initialize K species with different random latent means
        self.species = []
        per_species_pop = max(4, pop_size // n_species)
        for k in range(n_species):
            latent_init = np.random.randn(latent_dim) * 0.5
            sp = Species(latent_dim, latent_init, sigma0=sigma0, pop_size=per_species_pop)
            self.species.append(sp)

        self.generation = 0
        self.best_latent = None
        self.best_fitness = -np.inf
        self.best_genome = None
        self._env = None

    def decode(self, latent):
        h = np.tanh(latent @ self.decoder_W1 + self.decoder_b1)
        params = h @ self.decoder_W2 + self.decoder_b2
        return params * self.output_scale

    def _get_env(self):
        if self._env is None:
            self._env = gym.make(self.env_name)
        return self._env

    def _behavior_sig_quick(self, latent):
        params = self.decode(latent)
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
        # 1. Sample from all species
        all_latents = []
        for sp in self.species:
            lats = sp.sample()
            for l in lats:
                all_latents.append(l)
        all_latents = np.array(all_latents)

        # 2. Decode and evaluate
        fits = np.zeros(len(all_latents))
        for i, lat in enumerate(all_latents):
            params = self.decode(lat)
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = params.copy()
            d = g.to_genome_dict()
            fits[i] = max(fitness_fn(d), 1e-6)

        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        mean_fit = float(np.mean(fits))

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_latent = all_latents[best_idx].copy()
            params = self.decode(self.best_latent)
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = params.copy()
            self.best_genome = g

        # 3. Fitness shaping
        fits_std = np.std(fits)
        fits_mean = np.mean(fits)
        if (self.fitness_shaping_weight > 0 and
            fits_std < self.fitness_shaping_threshold * max(fits_mean, 1e-3)):
            sigs = []
            for lat in all_latents:
                s = self._behavior_sig_quick(lat)
                sigs.append(s)
            max_len = max(len(s) for s in sigs) if sigs else 1
            sigs_padded = np.zeros((len(sigs), max_len))
            for i, s in enumerate(sigs):
                sigs_padded[i, :len(s)] = s
            k = min(len(all_latents), 8)
            sample_idx = np.random.choice(len(all_latents), k, replace=False)
            diversity = np.zeros(len(all_latents))
            for i in range(len(all_latents)):
                diversity[i] = np.mean(np.linalg.norm(sigs_padded[i] - sigs_padded[sample_idx], axis=1))
            if diversity.max() > 0:
                diversity = diversity / diversity.max()
                fits = fits + self.fitness_shaping_weight * max(fits_mean, 1.0) * diversity

        # 4. Assign to species by behavior
        sigs = []
        for lat in all_latents:
            sigs.append(self._behavior_sig_quick(lat))
        max_len = max(len(s) for s in sigs) if sigs else 1
        sigs_padded = np.zeros((len(sigs), max_len))
        for i, s in enumerate(sigs):
            sigs_padded[i, :len(s)] = s

        species_sigs = []
        for sp in self.species:
            species_sigs.append(self._behavior_sig_quick(sp.center))
        species_sigs_padded = np.zeros((len(species_sigs), max_len))
        for i, s in enumerate(species_sigs):
            species_sigs_padded[i, :len(s)] = s

        dists = np.linalg.norm(sigs_padded[:, None, :] - species_sigs_padded[None, :, :], axis=2)
        assigned = np.argmin(dists, axis=1)

        # 5. Update each species
        for k, sp in enumerate(self.species):
            mask = assigned == k
            if mask.sum() < 2:
                continue
            sp.update(all_latents[mask], fits[mask])

        # 6. Merge / restart
        if self.generation > 0 and self.generation % 10 == 0:
            self._merge_and_restart()

        self.generation += 1
        return best_fit, mean_fit

    def _merge_and_restart(self):
        # Compute species behaviors
        species_beh = [self._behavior_sig_quick(sp.center) for sp in self.species]
        max_len = max(len(s) for s in species_beh) if species_beh else 1
        species_beh_padded = np.zeros((len(species_beh), max_len))
        for i, s in enumerate(species_beh):
            species_beh_padded[i, :len(s)] = s

        n = len(self.species)
        merged = set()
        for i in range(n):
            if i in merged:
                continue
            for j in range(i + 1, n):
                if j in merged:
                    continue
                d = np.linalg.norm(species_beh_padded[i] - species_beh_padded[j])
                if d < self.merge_threshold:
                    merged.add(j)
                    break
        if merged:
            keep = [i for i in range(n) if i not in merged]
            self.species = [self.species[i] for i in keep]

        for i, sp in enumerate(self.species):
            if sp.stagnation > self.stagnation_limit:
                new_latent = np.random.randn(self.latent_dim) * 0.5
                self.species[i] = Species(self.latent_dim, new_latent, sigma0=self.sigma0, pop_size=sp.pop_size)

        while len(self.species) < self.n_species:
            new_latent = np.random.randn(self.latent_dim) * 0.5
            self.species.append(Species(self.latent_dim, new_latent, sigma0=self.sigma0,
                                        pop_size=max(4, self.pop_size // self.n_species)))

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        params = self.decode(self.species[0].center)
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params
        return g.to_genome_dict()
