"""
MORPH v10 - v8 with aggressive diversity injection.

Round 10: NEAT beats MORPH on MountainCar because NEAT's initial population
has 30 RANDOM linear classifiers, some of which happen to push left when on
the right (the right behavior for MountainCar). MORPH v8 has K species centers
all initialized similarly (input->output on, rest off), so initial diversity
is limited.

v10 changes:
1. Each species starts with a DIFFERENT random subset of gates ON
   (not just input->output). This gives true topological diversity.
2. Initial sigma is larger (1.0) to spread out the population.
3. L0 pressure starts after a "growth phase" (first 10 generations) - lets
   topology grow first, then prunes.
4. Behavioral diversity is replaced by parameter diversity (cheaper, and
   more aligned with "different topologies = different species").

The key innovation: instead of NEAT's discrete add_node mutations, we have
CONTINUOUS topology diversity via random gate initialization. This is the
"modern" replacement for NEAT's complexification - we start with diverse
topologies and let selection pick the best.
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4
from src.morph_v8 import Species


class MorphV10:
    """v8 + random topology diversity + delayed L0 pressure."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=50, n_species=6, sigma0=1.0,
                 l0_pressure=0.02, l0_threshold=0.1,
                 l0_warmup_gens=10,
                 merge_threshold=0.5, stagnation_limit=15,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 init_topology_diversity=0.5,  # prob of random gate being ON in init
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.n_species = n_species
        self.pop_size = pop_size
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.l0_warmup_gens = l0_warmup_gens
        self.merge_threshold = merge_threshold
        self.stagnation_limit = stagnation_limit
        self.init_topology_diversity = init_topology_diversity

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

        # Initialize K species with DIVERSE random topologies
        self.species = []
        per_species_pop = max(4, pop_size // n_species)
        for k in range(n_species):
            center = self._make_diverse_init(template, k)
            sp = Species(self.dim, center.params, sigma0=sigma0, pop_size=per_species_pop)
            self.species.append(sp)

        self.generation = 0
        self.best_genome = None
        self.best_fitness = -np.inf
        self._env = None

    def _make_diverse_init(self, template, species_idx):
        """Create a genome with diverse random topology."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        # Each species gets a different random subset of gates ON
        rng = np.random.RandomState(species_idx * 1000 + 42)
        for idx, (a, b) in enumerate(g.candidate_conns):
            if a in self.input_ids and b in self.output_ids:
                # 70% chance of input->output being on
                g.params[idx] = 1.0 if rng.rand() < 0.7 else -1.0
                g.params[self.n_conns + idx] = rng.uniform(-1, 1)
            else:
                # Random gates: 30% chance of being on (diversity)
                g.params[idx] = 1.0 if rng.rand() < self.init_topology_diversity else -1.0
                g.params[self.n_conns + idx] = rng.uniform(-1, 1)
        return g

    def _get_env(self):
        if self._env is None:
            self._env = gym.make(self.env_name)
        return self._env

    def _behavior_signature(self, params):
        """Trajectory-based behavior."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params
        d = g.to_genome_dict()
        net = FeedForwardNet(d['nodes'], d['connections'], d['input_ids'], d['output_ids'])
        env = self._get_env()
        max_pos = -np.inf
        min_pos = np.inf
        trajectory = []
        obs, _ = env.reset(seed=self.seed_offset)
        for step in range(self.max_steps):
            logits = net.forward(obs)
            if self.is_continuous:
                a = np.tanh(logits)
                a = np.clip(a, env.action_space.low, env.action_space.high)
            else:
                a = int(np.argmax(logits))
            obs, r, terminated, truncated, _ = env.step(a)
            pos = float(obs[0])
            max_pos = max(max_pos, pos)
            min_pos = min(min_pos, pos)
            if (step % 10) == 0:
                trajectory.append(obs[:2])
            if terminated or truncated:
                break
        traj_arr = np.array(trajectory).flatten() if trajectory else np.zeros(2)
        max_len = 20
        if len(traj_arr) < max_len:
            traj_arr = np.concatenate([traj_arr, np.zeros(max_len - len(traj_arr))])
        else:
            traj_arr = traj_arr[:max_len]
        return np.concatenate([[max_pos, min_pos], traj_arr])

    def step(self, fitness_fn):
        # 1. Sample from all species
        all_samples = []
        for k, sp in enumerate(self.species):
            samples = sp.sample()
            for s in samples:
                all_samples.append(s)
        all_samples = np.array(all_samples)

        # 2. Evaluate
        fits = np.zeros(len(all_samples))
        for i, x in enumerate(all_samples):
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = x.copy()
            d = g.to_genome_dict()
            fits[i] = max(fitness_fn(d), 1e-6)

        # Track global best
        best_idx = int(np.argmax(fits))
        if fits[best_idx] > self.best_fitness:
            self.best_fitness = float(fits[best_idx])
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = all_samples[best_idx].copy()
            self.best_genome = g

        # 3. Compute behavior signatures
        sigs = np.array([self._behavior_signature(x) for x in all_samples])

        # 4. Assign to species by behavior
        species_sigs = np.array([self._behavior_signature(sp.center) for sp in self.species])
        dists = np.linalg.norm(sigs[:, None, :] - species_sigs[None, :, :], axis=2)
        assigned = np.argmin(dists, axis=1)

        # 5. Update each species
        for k, sp in enumerate(self.species):
            mask = assigned == k
            if mask.sum() < 2:
                continue
            sp.update(all_samples[mask], fits[mask])

            # L0 pressure (only after warmup)
            if self.l0_pressure > 0 and self.generation >= self.l0_warmup_gens:
                gate_logits = sp.center[:self.n_conns]
                weights_vec = sp.center[self.n_conns:2 * self.n_conns]
                active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
                gate_logits[active_small] -= self.l0_pressure
                sp.center[:self.n_conns] = gate_logits

        # 6. Merge / restart
        if self.generation > 0 and self.generation % 10 == 0:
            self._merge_and_restart()

        best_fit = float(np.max(fits))
        mean_fit = float(np.mean(fits))
        self.generation += 1
        return best_fit, mean_fit

    def _merge_and_restart(self):
        species_sigs = np.array([self._behavior_signature(sp.center) for sp in self.species])
        n = len(self.species)
        merged = set()
        for i in range(n):
            if i in merged:
                continue
            for j in range(i + 1, n):
                if j in merged:
                    continue
                d = np.linalg.norm(species_sigs[i] - species_sigs[j])
                if d < self.merge_threshold:
                    merged.add(j)
                    break
        if merged:
            keep = [i for i in range(n) if i not in merged]
            self.species = [self.species[i] for i in keep]

        for i, sp in enumerate(self.species):
            if sp.stagnation > self.stagnation_limit:
                # Restart with diverse init
                template = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
                new_center = self._make_diverse_init(template, np.random.randint(1000))
                self.species[i] = Species(self.dim, new_center.params, sigma0=1.0, pop_size=sp.pop_size)

        while len(self.species) < self.n_species:
            template = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            new_center = self._make_diverse_init(template, np.random.randint(1000))
            self.species.append(Species(self.dim, new_center.params, sigma0=1.0,
                                        pop_size=max(4, self.pop_size // self.n_species)))

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = self.species[0].center
        return g.to_genome_dict()
