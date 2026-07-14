"""
Reference NEAT implementation - Stanley & Miikkulainen 2002.
Used as baseline. Minimal but correct.

Key components:
- Innovation history (global) for marking new genes
- Speciation via k-means-like clustering on compatibility distance
- Explicit fitness sharing within species
- Mutation: weight perturbation, add_node, add_connection
- Crossover within species
"""
from __future__ import annotations
import numpy as np
import copy
from collections import defaultdict


# -------- Innovation history --------
class InnovationHistory:
    def __init__(self):
        self.history = {}  # (in_node, out_node) -> innovation_number
        self.next_innov = 0
        self.next_node = 0

    def get_innov(self, in_node, out_node):
        key = (in_node, out_node)
        if key not in self.history:
            self.history[key] = self.next_innov
            self.next_innov += 1
        return self.history[key]

    def new_node_id(self):
        nid = self.next_node
        self.next_node += 1
        return nid


# -------- Genome --------
def make_initial_genome(n_inputs, n_outputs, history, weight_init='uniform'):
    """Fully connected input->output, no hidden nodes."""
    nodes = []
    input_ids = list(range(n_inputs))
    output_ids = list(range(n_inputs, n_inputs + n_outputs))
    history.next_node = n_inputs + n_outputs

    for i in input_ids:
        nodes.append({'id': i, 'type': 'in', 'bias': 0.0, 'activation': 'identity'})
    for o in output_ids:
        nodes.append({'id': o, 'type': 'out', 'bias': 0.0, 'activation': 'tanh'})

    connections = []
    for i in input_ids:
        for o in output_ids:
            w = np.random.uniform(-1, 1) if weight_init == 'uniform' else np.random.randn()
            innov = history.get_innov(i, o)
            connections.append({'in': i, 'out': o, 'weight': w, 'enabled': True, 'innov': innov})

    return {
        'nodes': nodes,
        'connections': connections,
        'input_ids': input_ids,
        'output_ids': output_ids,
        'fitness': 0.0,
    }


def copy_genome(g):
    return {
        'nodes': [dict(n) for n in g['nodes']],
        'connections': [dict(c) for c in g['connections']],
        'input_ids': list(g['input_ids']),
        'output_ids': list(g['output_ids']),
        'fitness': g.get('fitness', 0.0),
    }


# -------- Mutations --------
def mutate_weights(g, rate=0.8, perturb_prob=0.9, perturb_scale=0.3):
    for c in g['connections']:
        if np.random.rand() < rate:
            if np.random.rand() < perturb_prob:
                c['weight'] += np.random.randn() * perturb_scale
            else:
                c['weight'] = np.random.uniform(-1, 1)
            c['weight'] = np.clip(c['weight'], -5, 5)
    for n in g['nodes']:
        if n['type'] != 'in' and np.random.rand() < rate * 0.5:
            if np.random.rand() < perturb_prob:
                n['bias'] += np.random.randn() * perturb_scale
            else:
                n['bias'] = np.random.uniform(-1, 1)
            n['bias'] = np.clip(n['bias'], -5, 5)


def mutate_add_node(g, history):
    """Split a random enabled connection into two with a new node."""
    enabled = [c for c in g['connections'] if c['enabled']]
    if not enabled:
        return
    c = enabled[np.random.randint(len(enabled))]
    c['enabled'] = False

    new_id = history.new_node_id()
    g['nodes'].append({'id': new_id, 'type': 'hidden', 'bias': 0.0, 'activation': 'tanh'})

    # in -> new (weight 1), new -> out (weight = old weight)
    innov1 = history.get_innov(c['in'], new_id)
    innov2 = history.get_innov(new_id, c['out'])
    g['connections'].append({'in': c['in'], 'out': new_id, 'weight': 1.0, 'enabled': True, 'innov': innov1})
    g['connections'].append({'in': new_id, 'out': c['out'], 'weight': c['weight'], 'enabled': True, 'innov': innov2})


def mutate_add_connection(g, history, max_tries=20):
    """Add a new connection between two random unconnected nodes."""
    if len(g['nodes']) < 2:
        return
    node_ids = [n['id'] for n in g['nodes']]
    existing = set((c['in'], c['out']) for c in g['connections'])

    for _ in range(max_tries):
        a = node_ids[np.random.randint(len(node_ids))]
        b = node_ids[np.random.randint(len(node_ids))]
        if a == b:
            continue
        # Don't allow inputs as targets, outputs as sources (no recurrence for FF)
        a_node = next(n for n in g['nodes'] if n['id'] == a)
        b_node = next(n for n in g['nodes'] if n['id'] == b)
        if a_node['type'] == 'in' and b_node['type'] == 'in':
            continue
        if a_node['type'] == 'out' and b_node['type'] == 'out':
            continue
        # No recurrence (simplified: ensure topological order by index)
        # We'll just allow any FF-friendly direction
        if b_node['type'] == 'in':
            continue
        if a_node['type'] == 'out':
            continue
        if (a, b) in existing:
            continue
        innov = history.get_innov(a, b)
        g['connections'].append({'in': a, 'out': b, 'weight': np.random.uniform(-1, 1), 'enabled': True, 'innov': innov})
        return


# -------- Compatibility distance (NEAT speciation) --------
def compatibility(g1, g2, c1=1.0, c2=1.0, c3=0.4):
    """NEAT's delta = c1*E/N + c2*D/N + c3*W_avg_diff."""
    conns1 = {c['innov']: c for c in g1['connections']}
    conns2 = {c['innov']: c for c in g2['connections']}
    innovs1 = set(conns1.keys())
    innovs2 = set(conns2.keys())

    max1 = max(innovs1) if innovs1 else 0
    max2 = max(innovs2) if innovs2 else 0
    cutoff = min(max1, max2)

    E = 0  # excess
    D = 0  # disjoint
    W_diff = 0.0
    matching = 0

    for i in innovs1 | innovs2:
        in1 = i in innovs1
        in2 = i in innovs2
        if in1 and in2:
            W_diff += abs(conns1[i]['weight'] - conns2[i]['weight'])
            matching += 1
        else:
            if i > cutoff:
                E += 1
            else:
                D += 1

    N = max(len(innovs1), len(innovs2))
    N = max(N, 1)
    W_avg = W_diff / max(matching, 1)
    return c1 * E / N + c2 * D / N + c3 * W_avg


# -------- Crossover --------
def crossover(g1, g2, history):
    """g1 should be the more fit parent."""
    if g2['fitness'] > g1['fitness']:
        g1, g2 = g2, g1
    conns1 = {c['innov']: c for c in g1['connections']}
    conns2 = {c['innov']: c for c in g2['connections']}
    all_innovs = sorted(set(conns1.keys()) | set(conns2.keys()))

    child_conns = []
    for i in all_innovs:
        in1 = i in conns1
        in2 = i in conns2
        if in1 and in2:
            c = copy.deepcopy(conns1[i]) if np.random.rand() < 0.5 else copy.deepcopy(conns2[i])
            # if either disabled, may stay disabled
            if not conns1[i]['enabled'] or not conns2[i]['enabled']:
                c['enabled'] = np.random.rand() < 0.75
            child_conns.append(c)
        elif in1:
            child_conns.append(copy.deepcopy(conns1[i]))
        else:
            # excess/disjoint from less-fit parent: skip in NEAT
            pass

    # Collect node IDs used
    used_nodes = set()
    for c in child_conns:
        used_nodes.add(c['in'])
        used_nodes.add(c['out'])
    # Always keep input/output nodes
    for nid in g1['input_ids'] + g1['output_ids']:
        used_nodes.add(nid)
    node_map = {n['id']: n for n in g1['nodes']}
    node_map2 = {n['id']: n for n in g2['nodes']}
    child_nodes = []
    for nid in used_nodes:
        n = node_map.get(nid, node_map2.get(nid))
        if n is not None:
            child_nodes.append(dict(n))

    return {
        'nodes': child_nodes,
        'connections': child_conns,
        'input_ids': list(g1['input_ids']),
        'output_ids': list(g1['output_ids']),
        'fitness': 0.0,
    }


# -------- NEAT algorithm --------
class NEAT:
    def __init__(self, n_inputs, n_outputs, pop_size=50,
                 compatibility_threshold=3.0,
                 c1=1.0, c2=1.0, c3=0.4,
                 weight_mut_rate=0.8,
                 node_mut_rate=0.03,
                 conn_mut_rate=0.05,
                 elitism=1,
                 asexual_prob=0.25,
                 target_species=8,
                 adjust_threshold=0.3):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.pop_size = pop_size
        self.compat_thresh = compatibility_threshold
        self.c1, self.c2, self.c3 = c1, c2, c3
        self.weight_mut_rate = weight_mut_rate
        self.node_mut_rate = node_mut_rate
        self.conn_mut_rate = conn_mut_rate
        self.elitism = elitism
        self.asexual_prob = asexual_prob
        self.target_species = target_species
        self.adjust_threshold = adjust_threshold

        self.history = InnovationHistory()
        self.population = [make_initial_genome(n_inputs, n_outputs, self.history) for _ in range(pop_size)]
        self.species = []
        self.generation = 0

    def speciate(self):
        species = []  # each: {'representative': genome, 'members': [idxs], 'avg_fitness': float}
        for i, g in enumerate(self.population):
            placed = False
            for s in species:
                if compatibility(g, s['representative'], self.c1, self.c2, self.c3) < self.compat_thresh:
                    s['members'].append(i)
                    placed = True
                    break
            if not placed:
                species.append({'representative': g, 'members': [i]})
        self.species = species

        # Adjust threshold toward target species count
        if self.adjust_threshold > 0 and len(species) > 0:
            if len(species) > self.target_species:
                self.compat_thresh += self.adjust_threshold * 0.5
            elif len(species) < self.target_species // 2 + 1:
                self.compat_thresh -= self.adjust_threshold * 0.5
            self.compat_thresh = max(0.5, self.compat_thresh)

    def compute_adjusted_fitness(self):
        """Explicit fitness sharing: divide each member's fitness by species size."""
        for s in self.species:
            n = len(s['members'])
            for i in s['members']:
                self.population[i]['fitness'] /= max(n, 1)

    def reproduce(self):
        # Compute total adjusted fitness
        total_adj = sum(self.population[i]['fitness'] for s in self.species for i in s['members'])
        if total_adj <= 0:
            total_adj = 1.0

        new_pop = []
        for s in self.species:
            members = sorted(s['members'], key=lambda i: -self.population[i]['fitness'])
            # Elitism: keep best
            for e in range(min(self.elitism, len(members))):
                new_pop.append(copy_genome(self.population[members[e]]))
            # Allocate offspring proportional to adjusted fitness
            species_adj = sum(self.population[i]['fitness'] for i in s['members'])
            n_offspring = int(round(species_adj / total_adj * self.pop_size))
            n_offspring = min(n_offspring, self.pop_size - len(new_pop) - (len(self.species) - self.species.index(s) - 1) * self.elitism)
            n_offspring = max(n_offspring, 0)
            for _ in range(n_offspring):
                if len(new_pop) >= self.pop_size:
                    break
                if np.random.rand() < self.asexual_prob or len(members) < 2:
                    # Asexual: copy + mutate
                    parent_idx = members[np.random.randint(len(members))]
                    child = copy_genome(self.population[parent_idx])
                else:
                    # Crossover
                    p1 = members[np.random.randint(len(members))]
                    p2 = members[np.random.randint(len(members))]
                    child = crossover(self.population[p1], self.population[p2], self.history)
                self._mutate(child)
                new_pop.append(child)
            if len(new_pop) >= self.pop_size:
                break

        # Fill any remaining slots with random mutated copies
        while len(new_pop) < self.pop_size:
            child = copy_genome(self.population[np.random.randint(self.pop_size)])
            self._mutate(child)
            new_pop.append(child)

        # Reset fitness
        for g in new_pop:
            g['fitness'] = 0.0

        self.population = new_pop[:self.pop_size]
        self.generation += 1

    def _mutate(self, g):
        mutate_weights(g, rate=self.weight_mut_rate)
        if np.random.rand() < self.node_mut_rate:
            mutate_add_node(g, self.history)
        if np.random.rand() < self.conn_mut_rate:
            mutate_add_connection(g, self.history)

    def step(self, fitness_fn):
        """Evaluate, speciate, reproduce. fitness_fn(genome) -> float."""
        raw_fits = []
        for g in self.population:
            g['fitness'] = max(fitness_fn(g), 1e-6)
            raw_fits.append(g['fitness'])
        best_raw = max(raw_fits)
        mean_raw = float(np.mean(raw_fits))
        self.speciate()
        self.compute_adjusted_fitness()
        self.reproduce()
        return best_raw, mean_raw
