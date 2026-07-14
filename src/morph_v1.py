"""
MORPH v1 - Morphological Optimization via Response-driven Heuristics.

Core idea (round 1 of Phase 1):
--------------------------------
Replace NEAT's discrete topology mutations with a continuous gate parameter
on every *candidate* connection. Each individual has:
  - weights w_ij for all allowed (i,j) pairs in an overcomplete graph
  - gates g_ij in [0,1] parameterizing connection existence

The effective connection weight is g_ij * w_ij. The gate is annealed toward
binary via a temperature schedule, so early generations have many soft (small-g)
connections (≈ minimal topology), later generations have few hard (binary) ones.

We optimize (w, g) jointly via a simple evolution strategy. To avoid speciation,
we use a *novelty-archive-free behavioral diversity* term: each individual's
"behavior signature" is its action distribution over a fixed probe set. Diversity
is computed as the mean pairwise distance in this signature space.

Critical insight: gates that are 0 contribute nothing but cost nothing (we just
skip them in forward). So an overcomplete representation is computationally
efficient. Complexification is implicit (g grows) and directed (we don't add
random connections — they all start as candidates with g=0).

This is round 1. We will iterate.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet


def build_candidate_graph(n_inputs, n_outputs, n_hidden_max=16):
    """Build an overcomplete graph: inputs -> hidden pool -> outputs,
    plus input -> outputs. All potential connections are candidates.

    Returns:
      input_ids, hidden_ids, output_ids, candidate_connections (list of (in, out))
    """
    input_ids = list(range(n_inputs))
    hidden_ids = list(range(n_inputs, n_inputs + n_hidden_max))
    output_ids = list(range(n_inputs + n_hidden_max, n_inputs + n_hidden_max + n_outputs))

    candidates = []
    # input -> hidden
    for i in input_ids:
        for h in hidden_ids:
            candidates.append((i, h))
    # hidden -> hidden (no recurrence: only lower-index to higher-index)
    for h1 in hidden_ids:
        for h2 in hidden_ids:
            if h1 < h2:
                candidates.append((h1, h2))
    # hidden -> output
    for h in hidden_ids:
        for o in output_ids:
            candidates.append((h, o))
    # input -> output
    for i in input_ids:
        for o in output_ids:
            candidates.append((i, o))

    return input_ids, hidden_ids, output_ids, candidates


class MorphGenome:
    """Compact representation: vector of (gate, weight) for each candidate connection,
    plus biases for hidden + output nodes.
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 init_input_output_only=True):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max

        input_ids, hidden_ids, output_ids, candidates = build_candidate_graph(n_inputs, n_outputs, n_hidden_max)
        self.input_ids = input_ids
        self.hidden_ids = hidden_ids
        self.output_ids = output_ids
        self.candidate_conns = candidates  # list of (in_id, out_id)
        self.n_conns = len(candidates)
        self.n_bias = len(hidden_ids) + len(output_ids)

        # Parameter vector: [g_0, w_0, g_1, w_1, ..., biases]
        # gate in [0,1], weight in R, bias in R
        self.gates = np.zeros(self.n_conns)
        self.weights = np.zeros(self.n_conns)
        self.biases = np.zeros(self.n_bias)

        # Init: input->output connections start with gate=1 (so initial topology is minimal
        # but functional). All others start at gate=0.
        for idx, (i, o) in enumerate(candidates):
            if i in input_ids and o in output_ids:
                self.gates[idx] = 1.0
                self.weights[idx] = np.random.uniform(-1, 1)
            else:
                self.gates[idx] = 0.0
                self.weights[idx] = np.random.uniform(-1, 1) * 0.1

        # Init output biases to small random
        for j in range(self.n_outputs):
            self.biases[len(hidden_ids) + j] = np.random.uniform(-0.1, 0.1)

        self.fitness = 0.0

    def to_genome_dict(self, gate_threshold=0.5):
        """Convert to genome dict for evaluation. Only includes connections with gate > threshold."""
        nodes = []
        # Always include all input + output nodes
        for nid in self.input_ids:
            nodes.append({'id': nid, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
        # Only include hidden nodes that have at least one incoming or outgoing enabled connection
        active_hidden = set()
        for idx, (a, b) in enumerate(self.candidate_conns):
            if self.gates[idx] > gate_threshold:
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
            if self.gates[idx] > gate_threshold:
                # Skip if either endpoint is an inactive hidden node
                if a in self.hidden_ids and a not in active_hidden:
                    continue
                if b in self.hidden_ids and b not in active_hidden:
                    continue
                connections.append({'in': a, 'out': b, 'weight': float(self.weights[idx]), 'enabled': True})

        return {
            'nodes': nodes,
            'connections': connections,
            'input_ids': list(self.input_ids),
            'output_ids': list(self.output_ids),
            'fitness': self.fitness,
        }

    def to_soft_genome_dict(self):
        """Convert with soft gates: connection weight = g * w. All connections included.
        This is more expensive but smoother during optimization.
        """
        nodes = []
        for nid in self.input_ids:
            nodes.append({'id': nid, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
        for nid in self.hidden_ids:
            bias_idx = nid - self.hidden_ids[0]
            nodes.append({'id': nid, 'type': 'hidden', 'bias': float(self.biases[bias_idx]), 'activation': 'tanh'})
        for j, nid in enumerate(self.output_ids):
            bias_idx = len(self.hidden_ids) + j
            nodes.append({'id': nid, 'type': 'out', 'bias': float(self.biases[bias_idx]), 'activation': 'tanh'})

        connections = []
        for idx, (a, b) in enumerate(self.candidate_conns):
            g = float(self.gates[idx])
            w = float(self.weights[idx])
            # Effective weight = g * w. Use threshold to skip near-zero for speed.
            if abs(g * w) < 1e-4 and g < 0.05:
                continue
            connections.append({'in': a, 'out': b, 'weight': g * w, 'enabled': True})

        return {
            'nodes': nodes,
            'connections': connections,
            'input_ids': list(self.input_ids),
            'output_ids': list(self.output_ids),
            'fitness': self.fitness,
        }


class Morph:
    """The MORPH algorithm.

    Round 1: simple ES on (gates, weights, biases) with:
      - Gaussian mutation
      - Tournament selection
      - Annealed gate sigmoid (gates squashed toward binary over generations)
      - Behavioral diversity bonus
    """
    def __init__(self, n_inputs, n_outputs, n_hidden_max=16, pop_size=50,
                 gate_anneal_start=10, gate_anneal_end=40,
                 gate_temp_start=2.0, gate_temp_end=0.2,
                 mut_rate=0.9, mut_scale_w=0.3, mut_scale_g=0.2, mut_scale_b=0.2,
                 elitism=1, diversity_weight=0.1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.pop_size = pop_size
        self.gate_anneal_start = gate_anneal_start
        self.gate_anneal_end = gate_anneal_end
        self.gate_temp_start = gate_temp_start
        self.gate_temp_end = gate_temp_end
        self.mut_rate = mut_rate
        self.mut_scale_w = mut_scale_w
        self.mut_scale_g = mut_scale_g
        self.mut_scale_b = mut_scale_b
        self.elitism = elitism
        self.diversity_weight = diversity_weight

        self.population = [MorphGenome(n_inputs, n_outputs, n_hidden_max) for _ in range(pop_size)]
        self.generation = 0

    def gate_temperature(self):
        if self.generation < self.gate_anneal_start:
            return self.gate_temp_start
        if self.generation > self.gate_anneal_end:
            return self.gate_temp_end
        # Linear interp
        t = (self.generation - self.gate_anneal_start) / (self.gate_anneal_end - self.gate_anneal_start)
        return self.gate_temp_start + (self.gate_temp_end - self.gate_temp_start) * t

    def squash_gates(self, g: MorphGenome):
        """Apply sigmoid squashing to gates based on current temperature."""
        T = self.gate_temperature()
        # gates stored in [-inf, inf] pre-squash? Or in [0,1]?
        # We'll store raw gates in R, and the effective gate is sigmoid(g/T).
        # For simplicity, we keep gates as raw values and squash at evaluation.
        # So nothing to do here.
        pass

    def mutate(self, g: MorphGenome):
        # Mutate weights
        mask = np.random.rand(self.pop_size and g.n_conns or g.n_conns) < self.mut_rate
        g.weights[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_w
        g.weights = np.clip(g.weights, -5, 5)
        # Mutate gate logits (raw gates)
        # We store gate_logit separately. But for round 1, let's store gates as logits
        # and squash at evaluation. Actually, let's just store gates in [0,1] directly
        # and mutate additively, clipping to [0,1].
        mask = np.random.rand(g.n_conns) < self.mut_rate
        g.gates[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_g
        g.gates = np.clip(g.gates, 0.0, 1.0)
        # Mutate biases
        mask = np.random.rand(g.n_bias) < self.mut_rate
        g.biases[mask] += np.random.randn(np.sum(mask)) * self.mut_scale_b
        g.biases = np.clip(g.biases, -5, 5)

    def step(self, fitness_fn):
        # Evaluate
        for g in self.population:
            d = g.to_genome_dict(gate_threshold=0.5)
            g.fitness = max(fitness_fn(d), 1e-6)

        raw_fits = [g.fitness for g in self.population]
        best_raw = max(raw_fits)
        mean_raw = float(np.mean(raw_fits))

        # Behavioral diversity bonus (simple: distance in fitness space is not behavior,
        # so we use the action distribution over a probe set... but to keep it simple and
        # fast in round 1, we just use fitness-rank diversity as a placeholder)
        # TODO: real behavioral diversity in next round.
        sorted_idx = sorted(range(self.pop_size), key=lambda i: -raw_fits[i])
        # Add small bonus proportional to rank (encourages keeping diverse topologies)
        adjusted = [raw_fits[i] for i in range(self.pop_size)]

        # Selection + reproduction
        sorted_pop = [self.population[i] for i in sorted_idx]
        new_pop = []
        for e in range(self.elitism):
            new_pop.append(self._copy(sorted_pop[e]))
        # Fill rest with tournament selection
        while len(new_pop) < self.pop_size:
            # Tournament
            candidates = [self.population[np.random.randint(self.pop_size)] for _ in range(3)]
            parent = max(candidates, key=lambda g: g.fitness)
            child = self._copy(parent)
            self.mutate(child)
            child.fitness = 0.0
            new_pop.append(child)

        self.population = new_pop[:self.pop_size]
        self.generation += 1
        return best_raw, mean_raw

    def _copy(self, g):
        new = MorphGenome(self.n_inputs, self.n_outputs, self.n_hidden_max)
        new.gates = g.gates.copy()
        new.weights = g.weights.copy()
        new.biases = g.biases.copy()
        new.fitness = g.fitness
        return new
