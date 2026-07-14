"""
Fixed-topology genetic algorithm baseline.
Single hidden layer (configurable), weight mutation + tournament selection.
No topology evolution - tests if just optimizing weights is enough.
"""
from __future__ import annotations
import numpy as np
import copy
from src.network import FeedForwardNet


def make_fixed_genome(n_inputs, n_outputs, n_hidden):
    input_ids = list(range(n_inputs))
    hidden_ids = list(range(n_inputs, n_inputs + n_hidden))
    output_ids = list(range(n_inputs + n_hidden, n_inputs + n_hidden + n_outputs))
    nodes = []
    for i in input_ids:
        nodes.append({'id': i, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
    for h in hidden_ids:
        nodes.append({'id': h, 'type': 'hidden', 'bias': 0.0, 'activation': 'tanh'})
    for o in output_ids:
        nodes.append({'id': o, 'type': 'out', 'bias': 0.0, 'activation': 'tanh'})
    connections = []
    for i in input_ids:
        for h in hidden_ids:
            connections.append({'in': i, 'out': h, 'weight': np.random.uniform(-1, 1), 'enabled': True})
    for h in hidden_ids:
        for o in output_ids:
            connections.append({'in': h, 'out': o, 'weight': np.random.uniform(-1, 1), 'enabled': True})
    return {
        'nodes': nodes,
        'connections': connections,
        'input_ids': input_ids,
        'output_ids': output_ids,
        'fitness': 0.0,
    }


def mutate_fixed(g, rate=0.8, scale=0.3):
    for c in g['connections']:
        if np.random.rand() < rate:
            c['weight'] += np.random.randn() * scale
            c['weight'] = np.clip(c['weight'], -5, 5)
    for n in g['nodes']:
        if n['type'] != 'in' and np.random.rand() < rate * 0.5:
            n['bias'] += np.random.randn() * scale
            n['bias'] = np.clip(n['bias'], -5, 5)


class FixedGA:
    """Tournament selection GA on a fixed topology."""
    def __init__(self, n_inputs, n_outputs, n_hidden=8, pop_size=50,
                 mut_rate=0.8, mut_scale=0.3, tournament_k=3, elitism=1):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.mut_scale = mut_scale
        self.k = tournament_k
        self.elitism = elitism
        self.population = [make_fixed_genome(n_inputs, n_outputs, n_hidden) for _ in range(pop_size)]
        self.generation = 0

    def step(self, fitness_fn):
        for g in self.population:
            g['fitness'] = max(fitness_fn(g), 1e-6)
        sorted_pop = sorted(self.population, key=lambda g: -g['fitness'])
        best = sorted_pop[0]['fitness']
        mean = float(np.mean([g['fitness'] for g in self.population]))
        new_pop = []
        for e in range(self.elitism):
            new_pop.append(copy.deepcopy(sorted_pop[e]))
        while len(new_pop) < self.pop_size:
            # tournament
            candidates = [self.population[np.random.randint(self.pop_size)] for _ in range(self.k)]
            parent = max(candidates, key=lambda g: g['fitness'])
            child = copy.deepcopy(parent)
            mutate_fixed(child, self.mut_rate, self.mut_scale)
            child['fitness'] = 0.0
            new_pop.append(child)
        self.population = new_pop[:self.pop_size]
        self.generation += 1
        return best, mean
