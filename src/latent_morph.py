"""
LATENT MORPH - Neuroevolution in compressed latent space.

Round 26: Completely new approach after round-25 review.

Core idea: Instead of evolving the full (gates, weights, biases) parameter
vector (178+ dims), evolve a COMPRESSED LATENT CODE (32 dims) that generates
the full network via a fixed decoder.

The decoder is a fixed random neural network mapping:
  latent_code (32-dim) → full_params (178+ dim)

The full_params are then interpreted as (gate_logits, weights, biases) as in
MORPH v14. The gate_logits are thresholded at 0 for hard eval.

Why this is fundamentally different from previous MORPH:
1. Search space is 32-dim, not 178+ dim → 5x smaller, faster CMA-ES
2. Full covariance CMA-ES is feasible (32x32 = 1024 entries, not 178^2 = 31684)
3. The decoder provides inductive bias: similar latent codes → similar networks
4. This is a genotype→phenotype mapping (biological metaphor)

Why this might work better:
- Lower dim → faster convergence
- Full CMA-ES → better optimization (captures correlations)
- Latent space naturally clusters → multi-modal search without explicit species

The decoder is fixed (not evolved). This keeps the algorithm simple and
avoids the "chicken-and-egg" problem of co-evolving decoder and latent codes.
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4


class LatentMorph:
    """Latent-space neuroevolution with continuous gate decoding."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 latent_dim=32,
                 pop_size=50, sigma0=0.5,
                 # Decoder config
                 decoder_hidden_dim=128, decoder_seed=42,
                 # MORPH params (applied after decoding)
                 l0_pressure=0.005, l0_threshold=0.05,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 # CMA-ES: use FULL covariance (feasible in low dim)
                 use_full_cma=True,
                 # Restarts
                 stagnation_limit=10, max_restarts=3,
                 # Fitness shaping
                 fitness_shaping_weight=0.15, fitness_shaping_threshold=0.5,
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.latent_dim = latent_dim
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.init_gate_logit_on = init_gate_logit_on
        self.init_gate_logit_off = init_gate_logit_off
        self.use_full_cma = use_full_cma
        self.stagnation_limit = stagnation_limit
        self.max_restarts = max_restarts
        self.fitness_shaping_weight = fitness_shaping_weight
        self.fitness_shaping_threshold = fitness_shaping_threshold

        self.env_name = env_name
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset
        self.is_continuous = is_continuous

        # Build template to get dim
        template = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = template.dim
        self.n_conns = template.n_conns
        self.input_ids = template.input_ids
        self.output_ids = template.output_ids

        # Build fixed decoder
        rng = np.random.RandomState(decoder_seed)
        self.decoder_W1 = rng.randn(latent_dim, decoder_hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.decoder_b1 = np.zeros(decoder_hidden_dim)
        self.decoder_W2 = rng.randn(decoder_hidden_dim, self.dim) * np.sqrt(2.0 / decoder_hidden_dim)
        self.decoder_b2 = np.zeros(self.dim)
        # Scale the output so decoded params have reasonable magnitude
        self.output_scale = 1.0

        # Initial latent code = 0 (decodes to decoder_b2, which is 0 → neutral)
        self.latent_mean = np.zeros(latent_dim)
        self._init_cma_state()

        self.generation = 0
        self.restart_count = 0
        self.stagnation = 0
        self.best_latent = None
        self.best_fitness = -np.inf
        self.best_genome = None
        self._env = None

    def _init_cma_state(self):
        self.sigma = self.sigma0
        if self.use_full_cma:
            # Full covariance CMA-ES
            self.C = np.eye(self.latent_dim)
            self.C_inv = np.eye(self.latent_dim)
            # For Cholesky-based sampling
            self.A = np.eye(self.latent_dim)  # C = A A^T
        else:
            self.C = np.ones(self.latent_dim)
        self.pc = np.zeros(self.latent_dim)
        self.ps = np.zeros(self.latent_dim)

        self.mu = max(1, self.pop_size // 2)
        weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        weights /= weights.sum()
        self.weights = weights
        self.mu_eff = 1.0 / np.sum(weights ** 2)

        self.cc = 4.0 / (self.latent_dim + 4)
        self.cs = (self.mu_eff + 2) / (self.latent_dim + self.mu_eff + 5)
        self.c1 = 2.0 / ((self.latent_dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((self.latent_dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (self.latent_dim + 1)) - 1) + self.cs
        self.chiN = np.sqrt(self.latent_dim) * (1 - 1.0 / (4 * self.latent_dim) + 1.0 / (21 * self.latent_dim ** 2))

        self.cma_generation = 0

    def decode(self, latent):
        """Decode latent code to full parameter vector."""
        h = np.tanh(latent @ self.decoder_W1 + self.decoder_b1)
        params = h @ self.decoder_W2 + self.decoder_b2
        return params * self.output_scale

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
        # 1. Sample latent codes
        latents = []
        for _ in range(self.pop_size):
            z = np.random.randn(self.latent_dim)
            if self.use_full_cma:
                x = self.latent_mean + self.sigma * (self.A @ z)
            else:
                x = self.latent_mean + self.sigma * np.sqrt(self.C) * z
            latents.append(x)
        latents = np.array(latents)

        # 2. Decode and evaluate
        fits = np.zeros(self.pop_size)
        for i, lat in enumerate(latents):
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
            self.best_latent = latents[best_idx].copy()
            params = self.decode(self.best_latent)
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = params.copy()
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
            for lat in latents:
                params = self.decode(lat)
                s = self._behavior_sig_quick(params)
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

        # 4. CMA-ES update
        order = np.argsort(-fits)
        sorted_latents = latents[order]
        old_mean = self.latent_mean.copy()
        self.latent_mean = sum(self.weights[i] * sorted_latents[i] for i in range(self.mu))

        y = (self.latent_mean - old_mean) / self.sigma
        if self.use_full_cma:
            # Full CMA-ES
            try:
                self.C_inv = np.linalg.inv(self.C)
                C_inv_sqrt = np.linalg.cholesky(self.C_inv).T  # upper triangular
                z = C_inv_sqrt @ y
            except np.linalg.LinAlgError:
                z = y / np.sqrt(np.diag(self.C))
        else:
            z = y / np.sqrt(self.C)

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * z
        hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (self.cma_generation + 1))) < (1.4 + 2.0 / (self.latent_dim + 1)) * self.chiN
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y

        delta = (1 - hsig) * self.cc * (2 - self.cc)
        if self.use_full_cma:
            term1 = self.c1 * np.outer(self.pc, self.pc)
            term2_sum = np.zeros_like(self.C)
            for i in range(self.mu):
                diff = sorted_latents[i] - old_mean
                term2_sum += self.weights[i] * np.outer(diff, diff)
            term2 = (self.cmu / self.sigma ** 2) * term2_sum
            self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
            # Add small diagonal for stability
            self.C += np.eye(self.latent_dim) * 1e-10
            # Update Cholesky factor
            try:
                self.A = np.linalg.cholesky(self.C).T  # C = A^T A
            except np.linalg.LinAlgError:
                # Reset
                self.C = np.eye(self.latent_dim)
                self.A = np.eye(self.latent_dim)
        else:
            term1 = self.c1 * (self.pc ** 2)
            term2 = self.cmu * sum(self.weights[i] * (sorted_latents[i] - old_mean) ** 2 for i in range(self.mu)) / self.sigma ** 2
            self.C = (1 - self.c1 - self.cmu + delta) * self.C + term1 + term2
            self.C = np.maximum(self.C, 1e-20)

        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.damps)
        self.sigma = max(self.sigma, 1e-12)

        # 5. Restart
        if (self.sigma < 1e-6 or self.stagnation >= self.stagnation_limit) and self.restart_count < self.max_restarts:
            self.restart_count += 1
            self.stagnation = 0
            # Restart with random latent
            self.latent_mean = np.random.randn(self.latent_dim) * 0.5
            self._init_cma_state()

        self.cma_generation += 1
        self.generation += 1
        return best_fit, mean_fit

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        params = self.decode(self.latent_mean)
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params
        return g.to_genome_dict()
