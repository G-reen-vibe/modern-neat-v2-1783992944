"""
MORPH v2 - Soft-gate version with proper annealing.

Round 2 changes from v1:
- Gates stored as logits (unbounded). Effective gate = sigmoid(g / T).
- T annealed from high (gates ≈ 0.5, many soft) to low (gates ≈ binary).
- During training, evaluation uses SOFT gates (effective weight = sigmoid(g/T) * w).
  This gives a smooth fitness signal w.r.t. gate logit changes.
- For final topology extraction, threshold gates > 0.5 (after squashing).
- Behavioral diversity: each individual's behavior signature = action distribution
  over a fixed probe set of observations. Diversity bonus = mean pairwise distance.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet


def build_candidate_graph(n_inputs, n_outputs, n_hidden_max=16):
    input_ids = list(range(n_inputs))
    hidden_ids = list(range(n_inputs, n_inputs + n_hidden_max))
    output_ids = list(range(n_inputs + n_hidden_max, n_inputs + n_hidden_max + n_outputs))
    candidates = []
    for i in input_ids:
        for h in hidden_ids:
            candidates.append((i, h))
    for h1 in hidden_ids:
        for h2 in hidden_ids:
            if h1 < h2:
                candidates.append((h1, h2))
    for h in hidden_ids:
        for o in output_ids:
            candidates.append((h, o))
    for i in input_ids:
        for o in output_ids:
            candidates.append((i, o))
    return input_ids, hidden_ids, output_ids, candidates


class MorphGenomeV2:
    """Gate logits + weights + biases. Soft eval via sigmoid squashing."""
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16, init_seed=None):
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

        # gate logits (unbounded). sigmoid(g/T) gives effective gate.
        self.gate_logits = np.zeros(self.n_conns)
        self.weights = np.zeros(self.n_conns)
        self.biases = np.zeros(self.n_bias)

        # Init: input->output logits high (so sigmoid ~= 1), all others low (sigmoid ~= 0)
        # sigmoid(-3) ≈ 0.05, sigmoid(3) ≈ 0.95
        for idx, (a, b) in enumerate(candidates):
            if a in input_ids and b in output_ids:
                self.gate_logits[idx] = 3.0
                self.weights[idx] = np.random.uniform(-1, 1)
            else:
                self.gate_logits[idx] = -3.0
                self.weights[idx] = np.random.uniform(-0.5, 0.5)
        for j in range(self.n_outputs):
            self.biases[len(hidden_ids) + j] = np.random.uniform(-0.1, 0.1)

        self.fitness = 0.0
        self.behavior_signature = None

    def effective_gates(self, T):
        """Returns the effective gate values given temperature T."""
        return 1.0 / (1.0 + np.exp(-self.gate_logits / max(T, 1e-6)))

    def to_soft_genome_dict(self, T, min_gate=0.05):
        """Convert to genome dict using soft gates. Skip gates below min_gate."""
        gates = self.effective_gates(T)
        nodes = []
        for nid in self.input_ids:
            nodes.append({'id': nid, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
        # Determine which hidden nodes are active (have at least one gate > min_gate)
        active_hidden = set()
        for idx, (a, b) in enumerate(self.candidate_conns):
            if gates[idx] > min_gate:
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
            if gates[idx] < min_gate:
                continue
            if a in self.hidden_ids and a not in active_hidden:
                continue
            if b in self.hidden_ids and b not in active_hidden:
                continue
            connections.append({'in': a, 'out': b,
                                'weight': float(gates[idx] * self.weights[idx]),
                                'enabled': True})
        return {
            'nodes': nodes, 'connections': connections,
            'input_ids': list(self.input_ids), 'output_ids': list(self.output_ids),
            'fitness': self.fitness,
        }

    def to_hard_genome_dict(self, T=1.0):
        """For final evaluation: threshold at sigmoid=0.5, i.e., gate_logit > 0."""
        nodes = []
        for nid in self.input_ids:
            nodes.append({'id': nid, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
        active_hidden = set()
        for idx, (a, b) in enumerate(self.candidate_conns):
            if self.gate_logits[idx] > 0:
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
            if self.gate_logits[idx] <= 0:
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


class MorphV2:
    """MORPH v2: soft-gate annealed ES with behavioral diversity.

    Key changes from v1:
    - gate_logits + sigmoid squashing with proper T annealing
    - Soft eval during training (smooth fitness signal)
    - Behavioral diversity: action distribution over probe set
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16, pop_size=50,
                 gate_temp_start=4.0, gate_temp_end=0.3,
                 anneal_start_gen=0, anneal_end_gen=40,
                 mut_rate=0.9, mut_scale_w=0.3, mut_scale_g=0.5, mut_scale_b=0.2,
                 elitism=1, diversity_weight=0.05,
                 n_probe_obs=20, probe_seed=42):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.pop_size = pop_size
        self.gate_temp_start = gate_temp_start
        self.gate_temp_end = gate_temp_end
        self.anneal_start_gen = anneal_start_gen
        self.anneal_end_gen = anneal_end_gen
        self.mut_rate = mut_rate
        self.mut_scale_w = mut_scale_w
        self.mut_scale_g = mut_scale_g
        self.mut_scale_b = mut_scale_b
        self.elitism = elitism
        self.diversity_weight = diversity_weight

        # Probe observations for behavioral diversity
        rng = np.random.RandomState(probe_seed)
        self.probe_obs = rng.uniform(-1, 1, size=(n_probe_obs, n_inputs)).astype(np.float32)

        self.population = [MorphGenomeV2(n_inputs, n_outputs, n_hidden_max) for _ in range(pop_size)]
        self.generation = 0
        self.best_genome = None

    def gate_temperature(self):
        if self.generation < self.anneal_start_gen:
            return self.gate_temp_start
        if self.generation > self.anneal_end_gen:
            return self.gate_temp_end
        t = (self.generation - self.anneal_start_gen) / max(self.anneal_end_gen - self.anneal_start_gen, 1)
        return self.gate_temp_start + (self.gate_temp_end - self.gate_temp_start) * t

    def mutate(self, g: MorphGenomeV2):
        mask = np.random.rand(g.n_conns) < self.mut_rate
        g.weights[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_w
        g.weights = np.clip(g.weights, -5, 5)
        mask = np.random.rand(g.n_conns) < self.mut_rate
        g.gate_logits[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_g
        mask = np.random.rand(g.n_bias) < self.mut_rate
        g.biases[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_b
        g.biases = np.clip(g.biases, -5, 5)

    def compute_behavior_signature(self, g: MorphGenomeV2, T):
        """Compute behavior signature: action distribution over probe observations."""
        d = g.to_soft_genome_dict(T, min_gate=0.05)
        net = FeedForwardNet(d['nodes'], d['connections'], d['input_ids'], d['output_ids'])
        # For each probe obs, compute softmax over outputs (for discrete)
        # If continuous, just return the action
        sigs = []
        for obs in self.probe_obs:
            logits = net.forward(obs)
            # Softmax with temperature
            z = logits - np.max(logits)
            ez = np.exp(z)
            p = ez / np.sum(ez)
            sigs.append(p)
        return np.concatenate(sigs)

    def step(self, fitness_fn):
        T = self.gate_temperature()

        # Evaluate fitness
        for g in self.population:
            d = g.to_soft_genome_dict(T, min_gate=0.05)
            g.fitness = max(fitness_fn(d), 1e-6)
            g.behavior_signature = self.compute_behavior_signature(g, T)

        raw_fits = [g.fitness for g in self.population]
        best_raw = max(raw_fits)
        mean_raw = float(np.mean(raw_fits))

        # Behavioral diversity bonus
        if self.diversity_weight > 0:
            sigs = np.array([g.behavior_signature for g in self.population])
            # Compute mean pairwise distance (sampled for efficiency)
            n = self.pop_size
            # For efficiency, compute distance to a random subset
            k = min(n, 10)
            sample_idx = np.random.choice(n, k, replace=False)
            diversity = np.zeros(n)
            for i in range(n):
                d = np.mean(np.linalg.norm(sigs[i] - sigs[sample_idx], axis=1))
                diversity[i] = d
            # Normalize and add bonus
            if diversity.max() > 0:
                diversity = diversity / diversity.max()
            adjusted = [raw_fits[i] * (1 + self.diversity_weight * diversity[i]) for i in range(n)]
        else:
            adjusted = raw_fits

        # Sort by adjusted fitness
        sorted_idx = sorted(range(self.pop_size), key=lambda i: -adjusted[i])
        new_pop = []
        for e in range(self.elitism):
            new_pop.append(self._copy(self.population[sorted_idx[e]]))
        while len(new_pop) < self.pop_size:
            # Tournament
            candidates = [self.population[np.random.randint(self.pop_size)] for _ in range(3)]
            parent = max(candidates, key=lambda g: adjusted[self.population.index(g)] if g in self.population else g.fitness)
            # The above is buggy since the copy may not be findable. Let's use indices:
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
        return self.best_genome.to_hard_genome_dict() if self.best_genome else None

    def _copy(self, g):
        new = MorphGenomeV2(self.n_inputs, self.n_outputs, self.n_hidden_max)
        new.gate_logits = g.gate_logits.copy()
        new.weights = g.weights.copy()
        new.biases = g.biases.copy()
        new.fitness = g.fitness
        return new
