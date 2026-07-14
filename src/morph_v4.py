"""
MORPH v4 - Antithetic gradient estimation (OpenAI-ES style) on (gates, weights, biases).

Round 4 key insight: instead of random mutation + selection (which is undirected
exploration of topology), use ANTITHETIC SAMPLING to estimate the gradient of
fitness w.r.t. each parameter (including gate logits). This makes topology growth
DIRECTED by the optimization landscape.

Algorithm:
  - Maintain a single "center" genome (gate_logits, weights, biases)
  - Each generation:
    1. Sample N antithetic pairs: (θ + ε·δ_i, θ - ε·δ_i), i=1..N
    2. Evaluate fitness of all 2N perturbed genomes
    3. Estimate gradient: g = (1/(2N)) Σ (fit(+δ_i) - fit(-δ_i)) * δ_i / ε
    4. Update center via Adam-like rule: θ += α * m / (√v + 1e-8)
  - Hard threshold for eval (logit > 0 => connection active)
  - L0-style sparsity: gate logits decay toward 0 (i.e., toward threshold)
    when the corresponding weight is small. This creates "use it or lose it" pressure.
  - Behavioral diversity NOT needed because ES naturally explores via sigma.

This is a fundamental algorithm: a unified ES that jointly optimizes topology
(through gate logits) and weights, with emergent complexification via sparsity pressure.

The key innovation vs OpenAI-ES: the parameters include gate_logits that have
an asymmetric role - they control topology. The L0 sparsity pressure is the
"minimal starting topology" principle from NEAT, made continuous.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph


class MorphGenomeV4:
    """Same as V3 genome but cleaner."""
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        input_ids, hidden_ids, output_ids, candidates = build_candidate_graph(n_inputs, n_outputs, n_hidden_max)
        self.input_ids = input_ids
        self.hidden_ids = hidden_ids
        self.output_ids = output_ids
        self.candidate_conns = candidates
        self.n_conns = len(candidates)
        self.n_bias = len(hidden_ids) + len(output_ids)
        self.dim = 2 * self.n_conns + self.n_bias  # gates, weights, biases

        # Flat parameter vector
        self.params = np.zeros(self.dim)
        # Initialize
        for idx, (a, b) in enumerate(candidates):
            if a in input_ids and b in output_ids:
                self.params[idx] = 1.0  # gate on
                self.params[self.n_conns + idx] = np.random.uniform(-1, 1)  # weight
            else:
                self.params[idx] = -1.0  # gate off
                self.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)
        for j in range(self.n_outputs):
            self.params[2 * self.n_conns + len(hidden_ids) + j] = np.random.uniform(-0.1, 0.1)

        self.fitness = 0.0

    @property
    def gate_logits(self):
        return self.params[:self.n_conns]

    @property
    def weights(self):
        return self.params[self.n_conns:2 * self.n_conns]

    @property
    def biases(self):
        return self.params[2 * self.n_conns:]

    def active_mask(self):
        return self.gate_logits > 0

    def to_genome_dict(self):
        mask = self.active_mask()
        nodes = []
        for nid in self.input_ids:
            nodes.append({'id': nid, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
        active_hidden = set()
        for idx, (a, b) in enumerate(self.candidate_conns):
            if mask[idx]:
                if a in self.hidden_ids:
                    active_hidden.add(a)
                if b in self.hidden_ids:
                    active_hidden.add(b)
        for nid in self.hidden_ids:
            if nid in active_hidden:
                bias_idx = nid - self.hidden_ids[0]
                nodes.append({'id': nid, 'type': 'hidden', 'bias': float(self.biases[bias_idx]), 'activation': 'tanh'})
        for j, nid in enumerate(self.output_ids):
            bias_idx = len(self.hidden_ids) + j
            nodes.append({'id': nid, 'type': 'out', 'bias': float(self.biases[bias_idx]), 'activation': 'tanh'})
        connections = []
        w = self.weights
        for idx, (a, b) in enumerate(self.candidate_conns):
            if not mask[idx]:
                continue
            if a in self.hidden_ids and a not in active_hidden:
                continue
            if b in self.hidden_ids and b not in active_hidden:
                continue
            connections.append({'in': a, 'out': b, 'weight': float(w[idx]), 'enabled': True})
        return {
            'nodes': nodes, 'connections': connections,
            'input_ids': list(self.input_ids), 'output_ids': list(self.output_ids),
            'fitness': self.fitness,
        }


class MorphV4:
    """OpenAI-ES style antithetic gradient estimation on (gates, weights, biases).

    Parameters: θ = (gate_logits, weights, biases) ∈ R^d
    Update: θ ← θ + α · m̂ / (√v̂ + ε), where m̂, v̂ are Adam moments of the gradient.

    Gradient estimate (OpenAI-ES):
      g = (1/(2N)) Σ_i (fit(θ+εδ_i) - fit(θ-εδ_i)) / ε · δ_i
      where δ_i ~ N(0, I) and σ is the perturbation std.

    Sparsity pressure on gates: each step, gate_logits decay slightly toward 0
    when |weight| is small. This is the L0 principle from v3, now applied as
    a continuous decay rather than a discrete push.

    Sigma adaptation: sigma is controlled by a simple rule based on recent improvement.
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=50, sigma=0.5, alpha=0.5,
                 beta1=0.9, beta2=0.999, eps=1e-8,
                 l0_pressure=0.02, l0_threshold=0.1,
                 sigma_adapt=True, sigma_adapt_rate=0.1,
                 adam_warmup=5,
                 rank_normalize=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        # pop_size must be even for antithetic pairs
        self.pop_size = pop_size if pop_size % 2 == 0 else pop_size + 1
        self.n_pairs = self.pop_size // 2
        self.sigma = sigma
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.sigma_adapt = sigma_adapt
        self.sigma_adapt_rate = sigma_adapt_rate
        self.adam_warmup = adam_warmup
        self.rank_normalize = rank_normalize

        self.center = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = self.center.dim

        # Adam state
        self.m = np.zeros(self.dim)
        self.v = np.zeros(self.dim)
        self.t = 0  # adam step

        # For sigma adaptation
        self.recent_best = -np.inf

        self.generation = 0
        self.best_genome = None
        self.best_fitness = -np.inf

    def step(self, fitness_fn):
        # 1. Sample antithetic perturbations
        # Use a shared seed for reproducibility within generation
        deltas = np.random.randn(self.n_pairs, self.dim)
        # Apply per-dimension sigma (currently uniform)
        # Build perturbed parameter sets
        center_params = self.center.params
        plus_params = center_params[None, :] + self.sigma * deltas
        minus_params = center_params[None, :] - self.sigma * deltas

        # 2. Evaluate fitness
        fits = np.zeros(2 * self.n_pairs)
        for i in range(self.n_pairs):
            g_plus = self._make_genome(plus_params[i])
            g_minus = self._make_genome(minus_params[i])
            d_plus = g_plus.to_genome_dict()
            d_minus = g_minus.to_genome_dict()
            f_plus = max(fitness_fn(d_plus), 1e-6)
            f_minus = max(fitness_fn(d_minus), 1e-6)
            fits[2 * i] = f_plus
            fits[2 * i + 1] = f_minus

        # Track best
        best_idx = np.argmax(fits)
        best_fit = float(fits[best_idx])
        mean_fit = float(np.mean(fits))

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            if best_idx % 2 == 0:
                self.best_genome = self._make_genome(plus_params[best_idx // 2])
            else:
                self.best_genome = self._make_genome(minus_params[best_idx // 2])

        # 3. Gradient estimate
        if self.rank_normalize:
            # Rank-normalize fitnesses (OpenAI-ES trick)
            order = np.argsort(fits)
            ranks = np.zeros_like(fits)
            ranks[order] = np.arange(1, len(fits) + 1) / len(fits) - 0.5  # [-0.5, 0.5]
            # Reshape into (n_pairs, 2): column 0 = plus ranks, column 1 = minus ranks
            plus_ranks = ranks[0::2]
            minus_ranks = ranks[1::2]
            # Gradient: g = (1/N) Σ (rank(+) - rank(-)) * δ_i / σ
            grad = np.mean(((plus_ranks - minus_ranks)[:, None] * deltas), axis=0) / self.sigma
        else:
            plus_fits = fits[0::2]
            minus_fits = fits[1::2]
            grad = np.mean(((plus_fits - minus_fits)[:, None] * deltas), axis=0) / self.sigma

        # 4. Adam update
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        # Per-parameter learning rate (Adam)
        update = self.alpha * m_hat / (np.sqrt(v_hat) + self.eps)
        # Apply warmup
        if self.t < self.adam_warmup:
            update *= self.t / self.adam_warmup
        self.center.params += update

        # 5. Sparsity pressure on gate_logits
        if self.l0_pressure > 0:
            gate_logits = self.center.params[:self.center.n_conns]
            weights = self.center.params[self.center.n_conns:2 * self.center.n_conns]
            # Active gates with small weight decay toward 0 (off)
            active_small = (gate_logits > 0) & (np.abs(weights) < self.l0_threshold)
            gate_logits[active_small] -= self.l0_pressure
            # Inactive gates with small weight are already off; no need to push further
            self.center.params[:self.center.n_conns] = gate_logits

        # 6. Sigma adaptation (simple: if best improved, increase sigma slightly; else decrease)
        if self.sigma_adapt:
            if best_fit > self.recent_best:
                self.sigma *= (1 + self.sigma_adapt_rate * 0.5)
            else:
                self.sigma *= (1 - self.sigma_adapt_rate)
            self.sigma = float(np.clip(self.sigma, 1e-4, 1.0))
        self.recent_best = max(self.recent_best, best_fit)

        # 7. Also decay center's gate_logits for active-small-weight connections
        # (Already done above)

        self.generation += 1
        return best_fit, mean_fit

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        return self.center.to_genome_dict()

    def _make_genome(self, params):
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params.copy()
        return g
