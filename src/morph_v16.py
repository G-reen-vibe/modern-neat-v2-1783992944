"""
MORPH v16 - v14 with restart averaging (SWA-style).

Round 36: Inspired by Stochastic Weight Averaging (SWA) in deep learning.
Instead of using only the best genome from the last restart, average the
best genomes from ALL restarts. This produces a more robust solution that
generalizes better.

The averaging is done in parameter space: avg_params = mean(best_params_i).
After averaging, the gate_logits are re-thresholded and the network is
re-evaluated.

This is a single-principle addition: the elite archive is used for averaging,
not just bookkeeping.
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4
from src.morph_v14 import MorphV14


class MorphV16(MorphV14):
    """v14 + restart averaging."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.restart_bests = []  # (params, fitness) from each restart

    def step(self, fitness_fn):
        result = super().step(fitness_fn)

        # Track best from each restart
        if self.restart_count > len(self.restart_bests):
            # A restart just happened
            if self.best_genome is not None:
                self.restart_bests.append((self.best_genome.params.copy(), self.best_fitness))

        return result

    def best_genome_dict(self):
        """Use averaged parameters if multiple restarts, else best."""
        if len(self.restart_bests) >= 2 and self.best_genome is not None:
            # Average the best params from all restarts
            all_params = [p for p, _ in self.restart_bests]
            all_params.append(self.best_genome.params.copy())
            avg_params = np.mean(all_params, axis=0)

            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = avg_params
            return g.to_genome_dict()
        elif self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        else:
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = self.center
            return g.to_genome_dict()
