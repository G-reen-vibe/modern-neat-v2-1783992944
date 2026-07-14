"""
Fast feed-forward neural network for neuroevolution.
Uses numpy for vectorized evaluation. Supports arbitrary topologies
(encoded as node list + connection list) and is JIT-friendly.

Genome encoding (shared across all algorithms in this project):
  nodes: list of dicts {id, type ('in'|'out'|'hidden'), bias, activation}
  connections: list of dicts {in, out, weight, enabled}

For RL we use a small fixed action-discrete policy: argmax over output logits,
or sample from softmax(logits) for stochastic policies.
"""
from __future__ import annotations
import numpy as np
from collections import defaultdict


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def identity(x):
    return x


ACTIVATIONS = {
    'tanh': tanh,
    'relu': relu,
    'sigmoid': sigmoid,
    'identity': identity,
}


class FeedForwardNet:
    """Evaluates a genome by topological sort. Inputs -> hidden layers (in topo order) -> outputs.
    Assumes the genome has been topologically sorted (or we sort here).
    """

    def __init__(self, nodes, connections, input_ids, output_ids):
        # nodes: list of dicts {id, bias, activation}
        # connections: list of dicts {in, out, weight, enabled}
        self.input_ids = list(input_ids)
        self.output_ids = list(output_ids)

        # Build node lookup
        self.node_map = {n['id']: n for n in nodes}
        self.input_set = set(input_ids)
        self.output_set = set(output_ids)

        # Build adjacency
        self.incoming = defaultdict(list)  # node_id -> list of (src_id, weight)
        for c in connections:
            if c.get('enabled', True):
                self.incoming[c['out']].append((c['in'], c['weight']))

        # Topological sort (Kahn's algorithm) over nodes that are reachable
        # Inputs first, then propagation
        self.eval_order = self._compute_eval_order()

    def _compute_eval_order(self):
        # Compute nodes reachable from inputs, in topological order
        # Use DFS-based topological sort
        visited = set()
        order = []
        temp = set()

        def dfs(nid):
            if nid in visited:
                return
            if nid in temp:
                # cycle - skip (shouldn't happen with FF nets)
                return
            temp.add(nid)
            for src, _ in self.incoming.get(nid, []):
                dfs(src)
            temp.discard(nid)
            visited.add(nid)
            order.append(nid)

        for nid in self.node_map:
            if nid not in visited:
                dfs(nid)

        # order is now in topological order (deps first)
        return order

    def forward(self, x):
        """x: numpy array of shape (n_inputs,). Returns numpy array of shape (n_outputs,)."""
        values = {}
        for i, nid in enumerate(self.input_ids):
            values[nid] = x[i]
            # inputs also get their bias and activation applied
            node = self.node_map.get(nid)
            if node is not None:
                values[nid] = ACTIVATIONS.get(node['activation'], identity)(values[nid] + node['bias'])

        for nid in self.eval_order:
            if nid in self.input_set:
                continue
            node = self.node_map.get(nid)
            if node is None:
                continue
            s = node['bias']
            for src, w in self.incoming.get(nid, []):
                s = s + w * values.get(src, 0.0)
            values[nid] = ACTIVATIONS.get(node['activation'], tanh)(s)

        return np.array([values.get(nid, 0.0) for nid in self.output_ids])

    def forward_batch(self, X):
        """X: (B, n_inputs). Returns (B, n_outputs)."""
        return np.stack([self.forward(x) for x in X])


def policy_action(net, obs, stochastic=False, temperature=1.0):
    """Discrete action policy. Returns action and log-prob."""
    logits = net.forward(obs)
    if stochastic:
        # softmax sampling
        z = logits / max(temperature, 1e-6)
        z = z - np.max(z)
        ez = np.exp(z)
        p = ez / np.sum(ez)
        action = int(np.random.choice(len(p), p=p))
        return action, float(np.log(p[action] + 1e-12))
    else:
        return int(np.argmax(logits)), 0.0


def continuous_action(net, obs):
    """Continuous action policy. Returns action (np.array of shape (1,))."""
    return np.tanh(net.forward(obs))
