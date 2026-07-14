"""
Optimized feed-forward network evaluation for neuroevolution.
Caches the topological order and adjacency structure for fast repeated evaluation.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict
from src.network import ACTIVATIONS, identity, tanh


class FastNet:
    """Optimized feed-forward net. Builds a computation order once, then evaluates fast."""

    def __init__(self, nodes, connections, input_ids, output_ids):
        self.input_ids = list(input_ids)
        self.output_ids = list(output_ids)
        self.input_set = set(input_ids)
        self.output_set = set(output_ids)

        self.node_map = {n['id']: n for n in nodes}
        self.node_ids = list(self.node_map.keys())

        # Build adjacency: for each node, list of (src_id, weight)
        self.incoming = defaultdict(list)
        for c in connections:
            if c.get('enabled', True):
                self.incoming[c['out']].append((c['in'], float(c['weight'])))

        # Topological order (DFS-based, deps first)
        # Only include nodes reachable from inputs or leading to outputs
        visited = set()
        order = []
        temp = set()

        def dfs(nid):
            if nid in visited:
                return
            if nid in temp:
                return  # cycle
            temp.add(nid)
            for src, _ in self.incoming.get(nid, []):
                dfs(src)
            temp.discard(nid)
            visited.add(nid)
            order.append(nid)

        for nid in self.node_ids:
            if nid not in visited:
                dfs(nid)

        self.eval_order = [nid for nid in order if nid not in self.input_set]

        # Pre-compute input node biases/activations (inputs typically have bias 0, identity activation)
        self.input_biases = np.array([self.node_map[nid].get('bias', 0.0) for nid in self.input_ids])
        self.input_acts = [self.node_map[nid].get('activation', 'identity') for nid in self.input_ids]

        # For non-input nodes, store bias and activation
        self.node_biases = {}
        self.node_acts = {}
        for nid in self.eval_order:
            n = self.node_map.get(nid)
            if n is not None:
                self.node_biases[nid] = float(n.get('bias', 0.0))
                self.node_acts[nid] = n.get('activation', 'tanh')

        # Convert incoming to arrays for vectorized weighted sum
        # For each node in eval_order: arrays of (src_id, weight)
        self.node_sources = {}
        self.node_weights = {}
        for nid in self.eval_order:
            if nid in self.incoming:
                srcs, ws = zip(*self.incoming[nid])
                self.node_sources[nid] = list(srcs)
                self.node_weights[nid] = np.array(ws)
            else:
                self.node_sources[nid] = []
                self.node_weights[nid] = np.array([])

    def forward(self, x):
        values = {}
        for i, nid in enumerate(self.input_ids):
            values[nid] = x[i]
            # Inputs may have a bias and activation
            b = self.input_biases[i]
            if b != 0.0:
                values[nid] = values[nid] + b
            act = self.input_acts[i]
            if act != 'identity':
                values[nid] = ACTIVATIONS[act](values[nid])

        for nid in self.eval_order:
            s = self.node_biases.get(nid, 0.0)
            srcs = self.node_sources[nid]
            if srcs:
                ws = self.node_weights[nid]
                # Manual sum (small overhead, fast enough)
                for j, src in enumerate(srcs):
                    s += ws[j] * values.get(src, 0.0)
            act = self.node_acts.get(nid, 'tanh')
            values[nid] = ACTIVATIONS[act](s)

        return np.array([values.get(nid, 0.0) for nid in self.output_ids])
