"""
MORPH v3 - Hard-eval with annealed flip-sensitivity.

Round 3 change: instead of soft gates (which cause train/eval mismatch), we use
HARD gates (threshold at logit > 0) throughout. The temperature controls how
sensitive gates are to mutation, not the squashing.

Specifically:
- gate_logits stored as before
- effective_gate = 1 if logit > 0 else 0 (HARD threshold)
- mutation: logit += N(0, mut_scale_g)
- annealing: the *initial logit* for newly "active" connections starts near 0
  (easy to flip), and gradually increases in magnitude as generations progress
  (locking in stable connections)

Actually, a cleaner formulation: temperature controls the mutation scale itself.
Early generations: large mutations (exploration of topology)
Late generations: small mutations (refinement of weights)

Combined with the gate_logit framework: when a logit is near 0, it's "probationary"
(easy to flip on/off). When it's far from 0, it's "committed".

We add a key innovation: an L0-style "use it or lose it" pressure. Connections
with logit > 0 (active) but small |weight| get their logit pushed down (toward off).
This naturally prunes unused connections.

Behavioral diversity is kept from v2.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph


class MorphGenomeV3:
    """Hard-gate version. Effective gate = 1 if logit > 0 else 0."""
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

        self.gate_logits = np.zeros(self.n_conns)
        self.weights = np.zeros(self.n_conns)
        self.biases = np.zeros(self.n_bias)

        # Init: input->output logits positive (on), all others negative (off)
        for idx, (a, b) in enumerate(candidates):
            if a in input_ids and b in output_ids:
                self.gate_logits[idx] = 1.0
                self.weights[idx] = np.random.uniform(-1, 1)
            else:
                self.gate_logits[idx] = -1.0
                self.weights[idx] = np.random.uniform(-0.5, 0.5)
        for j in range(self.n_outputs):
            self.biases[len(hidden_ids) + j] = np.random.uniform(-0.1, 0.1)

        self.fitness = 0.0
        self.behavior_signature = None

    def active_mask(self):
        return self.gate_logits > 0

    def to_genome_dict(self):
        """Hard threshold. Only include connections with logit > 0."""
        mask = self.active_mask()
        nodes = []
        for nid in self.input_ids:
            nodes.append({'id': nid, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
        # Active hidden nodes
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
        for idx, (a, b) in enumerate(self.candidate_conns):
            if not mask[idx]:
                continue
            if a in self.hidden_ids and a not in active_hidden:
                continue
            if b in self.hidden_ids and b not in active_hidden:
                continue
            connections.append({'in': a, 'out': b, 'weight': float(self.weights[idx]), 'enabled': True})
        return {
            'nodes': nodes, 'connections': connections,
            'input_ids': list(self.input_ids), 'output_ids': list(self.output_ids),
            'fitness': self.fitness,
        }


class MorphV3:
    """MORPH v3: Hard-gate evaluation + annealed mutation scale + L0 pruning pressure.

    Key innovations:
    - Hard thresholding (no train/eval mismatch)
    - Mutation scale anneals: large early (topology exploration), small late (refinement)
    - L0 pruning: active connections with small |weight| get logit pushed down
      (use it or lose it)
    - Behavioral diversity preserved
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16, pop_size=50,
                 mut_scale_start=0.8, mut_scale_end=0.15,
                 anneal_start_gen=0, anneal_end_gen=40,
                 mut_rate=0.9, mut_scale_w=0.3, mut_scale_b=0.2,
                 mut_scale_g_start=1.0, mut_scale_g_end=0.1,
                 l0_pressure=0.05, l0_threshold=0.1,
                 elitism=1, diversity_weight=0.05,
                 n_probe_obs=20, probe_seed=42):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.mut_scale_w = mut_scale_w
        self.mut_scale_b = mut_scale_b
        self.mut_scale_g_start = mut_scale_g_start
        self.mut_scale_g_end = mut_scale_g_end
        self.anneal_start_gen = anneal_start_gen
        self.anneal_end_gen = anneal_end_gen
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.elitism = elitism
        self.diversity_weight = diversity_weight

        rng = np.random.RandomState(probe_seed)
        self.probe_obs = rng.uniform(-1, 1, size=(n_probe_obs, n_inputs)).astype(np.float32)

        self.population = [MorphGenomeV3(n_inputs, n_outputs, n_hidden_max) for _ in range(pop_size)]
        self.generation = 0
        self.best_genome = None

    def gate_mut_scale(self):
        if self.generation < self.anneal_start_gen:
            return self.mut_scale_g_start
        if self.generation > self.anneal_end_gen:
            return self.mut_scale_g_end
        t = (self.generation - self.anneal_start_gen) / max(self.anneal_end_gen - self.anneal_start_gen, 1)
        return self.mut_scale_g_start + (self.mut_scale_g_end - self.mut_scale_g_start) * t

    def mutate(self, g: MorphGenomeV3):
        # Weight mutation
        mask = np.random.rand(g.n_conns) < self.mut_rate
        g.weights[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_w
        g.weights = np.clip(g.weights, -5, 5)
        # Gate logit mutation (with annealed scale)
        ms = self.gate_mut_scale()
        mask = np.random.rand(g.n_conns) < self.mut_rate
        g.gate_logits[mask] += np.random.randn(np.sum(mask)) * ms
        # Bias mutation
        mask = np.random.rand(g.n_bias) < self.mut_rate
        g.biases[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_b
        g.biases = np.clip(g.biases, -5, 5)

        # L0 pruning pressure: active connections with small weight get pushed off
        if self.l0_pressure > 0:
            active = g.active_mask()
            small_active = active & (np.abs(g.weights) < self.l0_threshold)
            # Push their logits down (toward off)
            g.gate_logits[small_active] -= self.l0_pressure

    def compute_behavior_signature(self, g: MorphGenomeV3):
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
        # Evaluate fitness (HARD eval)
        for g in self.population:
            d = g.to_genome_dict()
            g.fitness = max(fitness_fn(d), 1e-6)
            g.behavior_signature = self.compute_behavior_signature(g)

        raw_fits = [g.fitness for g in self.population]
        best_raw = max(raw_fits)
        mean_raw = float(np.mean(raw_fits))

        # Behavioral diversity bonus
        if self.diversity_weight > 0:
            sigs = np.array([g.behavior_signature for g in self.population])
            n = self.pop_size
            k = min(n, 10)
            sample_idx = np.random.choice(n, k, replace=False)
            diversity = np.zeros(n)
            for i in range(n):
                d = np.mean(np.linalg.norm(sigs[i] - sigs[sample_idx], axis=1))
                diversity[i] = d
            if diversity.max() > 0:
                diversity = diversity / diversity.max()
            adjusted = np.array([raw_fits[i] * (1 + self.diversity_weight * diversity[i]) for i in range(n)])
        else:
            adjusted = np.array(raw_fits)

        # Selection
        sorted_idx = np.argsort(-adjusted)
        new_pop = []
        for e in range(self.elitism):
            new_pop.append(self._copy(self.population[sorted_idx[e]]))
        while len(new_pop) < self.pop_size:
            cand_idx = [np.random.randint(self.pop_size) for _ in range(3)]
            parent_idx = max(cand_idx, key=lambda i: adjusted[i])
            child = self._copy(self.population[parent_idx])
            self.mutate(child)
            child.fitness = 0.0
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self.best_genome = max(self.population, key=lambda g: g.fitness)
        self.generation += 1
        return best_raw, mean_raw

    def best_genome_dict(self):
        return self.best_genome.to_genome_dict() if self.best_genome else None

    def _copy(self, g):
        new = MorphGenomeV3(self.n_inputs, self.n_outputs, self.n_hidden_max)
        new.gate_logits = g.gate_logits.copy()
        new.weights = g.weights.copy()
        new.biases = g.biases.copy()
        new.fitness = g.fitness
        return new
