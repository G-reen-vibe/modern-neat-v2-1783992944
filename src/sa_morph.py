"""
SA-MORPH (Surrogate-Assisted MORPH) - Round 51+ new direction.

Core idea: Train a surrogate model (MLP) that predicts fitness from genome
parameters. Use it to screen many candidates cheaply, evaluate only top-K
with the real environment.

Algorithm:
  1. Maintain CMA-ES center (as in MORPH v14)
  2. Maintain surrogate MLP: params → predicted_fitness
  3. Each generation:
     a. Sample N_candidates (e.g., 200) from CMA-ES
     b. Predict fitness for all N_candidates using surrogate
     c. Select top-K (e.g., 30) by predicted fitness
     d. Evaluate top-K with real environment
     e. Add (params, real_fitness) to training data
     f. Periodically retrain surrogate
     g. Update CMA-ES using the K real evaluations

This gives:
- 200 candidates explored per generation (vs 30 for plain CMA-ES)
- Only 30 real env evaluations per generation (same as CMA-ES)
- Surrogate learns the fitness landscape → better exploration

The surrogate is a simple 2-layer MLP trained with SGD on (params, fitness) pairs.
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v4 import MorphGenomeV4


class SurrogateMLP:
    """Simple 2-layer MLP that predicts fitness from params."""
    def __init__(self, input_dim, hidden_dim=64, lr=0.001, epochs=50, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
        self.X_train = []
        self.y_train = []

    def predict(self, X):
        """X: (N, input_dim) → predictions (N,)."""
        h = np.tanh(X @ self.W1 + self.b1)
        out = h @ self.W2 + self.b2
        return out.flatten()

    def add_data(self, X, y):
        """Add training data. X: (N, dim), y: (N,)."""
        for i in range(len(X)):
            self.X_train.append(X[i])
            self.y_train.append(y[i])
        # Keep only last 1000 samples
        if len(self.X_train) > 1000:
            self.X_train = self.X_train[-1000:]
            self.y_train = self.y_train[-1000:]

    def train(self):
        """Train the MLP on accumulated data."""
        if len(self.X_train) < 10:
            return
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        # Normalize y
        y_mean = y.mean()
        y_std = y.std() + 1e-8
        y_norm = (y - y_mean) / y_std

        # Also normalize X (per-feature)
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std

        n = len(X)
        for epoch in range(self.epochs):
            # Shuffle
            perm = np.random.permutation(n)
            X_shuf = X_norm[perm]
            y_shuf = y_norm[perm]
            # Mini-batch SGD
            for i in range(0, n, self.batch_size):
                X_batch = X_shuf[i:i+self.batch_size]
                y_batch = y_shuf[i:i+self.batch_size]
                # Forward
                h = np.tanh(X_batch @ self.W1 + self.b1)
                out = h @ self.W2 + self.b2
                pred = out.flatten()
                # Loss: MSE
                err = pred - y_batch
                # Backward
                d_out = (2.0 / len(y_batch)) * err[:, None]
                d_W2 = h.T @ d_out
                d_b2 = d_out.sum(axis=0)
                d_h = d_out @ self.W2.T
                d_z = d_h * (1 - h ** 2)  # tanh derivative
                d_W1 = X_batch.T @ d_z
                d_b1 = d_z.sum(axis=0)
                # Update
                self.W1 -= self.lr * d_W1
                self.b1 -= self.lr * d_b1
                self.W2 -= self.lr * d_W2
                self.b2 -= self.lr * d_b2

        # Store normalization
        self.X_mean = X_mean
        self.X_std = X_std
        self.y_mean = y_mean
        self.y_std = y_std

    def predict_normalized(self, X):
        """Predict with normalization."""
        if not hasattr(self, 'X_mean'):
            # Not trained yet, return zeros
            return np.zeros(len(X))
        X_norm = (X - self.X_mean) / self.X_std
        return self.predict(X_norm) * self.y_std + self.y_mean


class SAMorph:
    """Surrogate-Assisted MORPH."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=30, n_candidates=200,
                 sigma0=1.5,
                 l0_pressure=0.005, l0_threshold=0.05,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 init_topology_diversity=0.3,
                 stagnation_limit=8, sigma_restart_threshold=1e-3,
                 max_restarts=5, pop_doubling=False,  # disable for surrogate
                 surrogate_hidden_dim=64,
                 surrogate_retrain_interval=5,
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.pop_size = pop_size  # K: real evals per gen
        self.n_candidates = n_candidates  # N: surrogate evals per gen
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
        self.surrogate_retrain_interval = surrogate_retrain_interval

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

        # Diverse init
        self.center = self._make_diverse_init()
        self._init_cma_state()

        # Surrogate
        self.surrogate = SurrogateMLP(self.dim, hidden_dim=surrogate_hidden_dim)

        self.generation = 0
        self.restart_count = 0
        self.stagnation = 0
        self.best_genome = None
        self.best_fitness = -np.inf
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

    def step(self, fitness_fn):
        # 1. Sample N candidates from CMA-ES
        candidates = []
        for _ in range(self.n_candidates):
            z = np.random.randn(self.dim)
            x = self.center + self.sigma * np.sqrt(self.C) * z
            candidates.append(x)
        candidates = np.array(candidates)

        # 2. Predict fitness using surrogate (if we have enough data)
        if len(self.surrogate.X_train) >= 10:
            self.surrogate.train() if self.generation % self.surrogate_retrain_interval == 0 else None
            predicted = self.surrogate.predict_normalized(candidates)
            # Select top-K by predicted fitness
            top_k_idx = np.argsort(-predicted)[:self.pop_size]
        else:
            # Not enough data: select random K
            top_k_idx = np.random.choice(self.n_candidates, self.pop_size, replace=False)

        selected = candidates[top_k_idx]

        # 3. Evaluate top-K with real environment
        fits = np.zeros(self.pop_size)
        for i, x in enumerate(selected):
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = x.copy()
            d = g.to_genome_dict()
            fits[i] = max(fitness_fn(d), 1e-6)

        # 4. Add to surrogate training data
        self.surrogate.add_data(selected, fits)

        # 5. Track best
        best_idx = int(np.argmax(fits))
        best_fit = float(fits[best_idx])
        mean_fit = float(np.mean(fits))

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = selected[best_idx].copy()
            self.best_genome = g
            self.stagnation = 0
        else:
            self.stagnation += 1

        # 6. Sep-CMA-ES update using the K real evaluations
        order = np.argsort(-fits)
        sorted_samples = selected[order]
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

        # 7. L0 pressure
        if self.l0_pressure > 0:
            gate_logits = self.center[:self.n_conns]
            weights_vec = self.center[self.n_conns:2 * self.n_conns]
            active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
            gate_logits[active_small] -= self.l0_pressure
            self.center[:self.n_conns] = gate_logits

        # 8. Restart
        if (self.sigma < self.sigma_restart_threshold or self.stagnation >= self.stagnation_limit) and self.restart_count < self.max_restarts:
            self.restart_count += 1
            self.stagnation = 0
            if self.pop_doubling:
                self.pop_size = min(self.pop_size * 2, 200)
            self.center = self._make_diverse_init()
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
