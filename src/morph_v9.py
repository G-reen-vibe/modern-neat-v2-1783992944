"""
MORPH v9 - Multi-species + stochastic policy with entropy bonus.

Round 9: tackle exploration directly with stochastic policies. The deterministic
argmax policy can't escape local optima on MountainCar (which requires
oscillation). v9 samples actions from softmax(logits/T) during training, with:
  - T annealed from 2.0 to 0.5 (always some stochasticity)
  - Entropy bonus added to fitness (encourages diverse action distributions)
  - Final eval: deterministic (argmax) for fair comparison

v9 also keeps the multi-species structure from v8, but uses trajectory-based
behavior characterization (max position reached) instead of probe-obs action
distribution. This is more meaningful for exploration-heavy envs.

Implementation note: v9 does its own evaluation (doesn't use fitness_fn passed
in) because it needs to compute entropy and trajectory.
"""
from __future__ import annotations
import numpy as np
import copy
import gymnasium as gym
from src.network import FeedForwardNet
from src.morph_v2 import build_candidate_graph
from src.morph_v4 import MorphGenomeV4
from src.morph_v8 import Species


class MorphV9:
    """v8 + stochastic policy + entropy bonus + trajectory-based behavior."""

    def __init__(self, n_inputs, n_outputs, n_hidden_max=16,
                 pop_size=50, n_species=4, sigma0=0.5,
                 l0_pressure=0.02, l0_threshold=0.1,
                 train_temp_start=2.0, train_temp_end=0.5,
                 entropy_weight=0.1,
                 merge_threshold=0.5, stagnation_limit=15,
                 init_gate_logit_on=1.0, init_gate_logit_off=-1.0,
                 env_name=None, max_steps=200, n_episodes=3, seed_offset=0,
                 is_continuous=False):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden_max = n_hidden_max
        self.n_species = n_species
        self.pop_size = pop_size
        self.l0_pressure = l0_pressure
        self.l0_threshold = l0_threshold
        self.merge_threshold = merge_threshold
        self.stagnation_limit = stagnation_limit
        self.train_temp_start = train_temp_start
        self.train_temp_end = train_temp_end
        self.entropy_weight = entropy_weight

        self.env_name = env_name
        self.max_steps = max_steps
        self.n_episodes = n_episodes
        self.seed_offset = seed_offset
        self.is_continuous = is_continuous

        template = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
        self.dim = template.dim
        self.n_conns = template.n_conns

        self.species = []
        per_species_pop = max(4, pop_size // n_species)
        for k in range(n_species):
            center = MorphGenomeV4(n_inputs, n_outputs, n_hidden_max)
            for idx, (a, b) in enumerate(center.candidate_conns):
                if a in center.input_ids and b in center.output_ids:
                    center.params[idx] = init_gate_logit_on + np.random.uniform(-0.5, 0.5)
                    center.params[self.n_conns + idx] = np.random.uniform(-1, 1)
                else:
                    center.params[idx] = init_gate_logit_off + np.random.uniform(-0.5, 0.5)
                    center.params[self.n_conns + idx] = np.random.uniform(-0.5, 0.5)
            sp = Species(self.dim, center.params, sigma0=sigma0, pop_size=per_species_pop)
            self.species.append(sp)

        self.generation = 0
        self.best_genome = None
        self.best_fitness = -np.inf
        self._env = None

    def _get_env(self):
        if self._env is None:
            self._env = gym.make(self.env_name)
        return self._env

    def train_temp(self):
        # Anneal over ~30 generations
        end_gen = 30
        if self.generation >= end_gen:
            return self.train_temp_end
        t = self.generation / end_gen
        return self.train_temp_start + (self.train_temp_end - self.train_temp_start) * t

    def _eval_stochastic(self, params, T):
        """Evaluate with stochastic policy. Returns (raw_reward, entropy, behavior_signature)."""
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = params
        d = g.to_genome_dict()
        net = FeedForwardNet(d['nodes'], d['connections'], d['input_ids'], d['output_ids'])
        env = self._get_env()

        total_reward = 0.0
        total_entropy = 0.0
        total_steps = 0
        # Behavior: max abs position reached (or first 10 obs dims averaged)
        max_pos_reached = -np.inf
        min_pos_reached = np.inf
        # Capture trajectory (downsampled)
        trajectory = []

        for ep in range(self.n_episodes):
            obs, _ = env.reset(seed=self.seed_offset + ep)
            for step in range(self.max_steps):
                logits = net.forward(obs)
                if self.is_continuous:
                    a = np.tanh(logits)
                    a = np.clip(a, env.action_space.low, env.action_space.high)
                    ent = 0.0  # not well-defined for continuous
                else:
                    z = logits / max(T, 1e-3)
                    z = z - np.max(z)
                    ez = np.exp(z)
                    p = ez / np.sum(ez)
                    a = int(np.random.choice(len(p), p=p))
                    ent = -np.sum(p * np.log(p + 1e-12))
                obs, r, terminated, truncated, _ = env.step(a)
                total_reward += r
                total_entropy += ent
                total_steps += 1
                # Track position (assume first dim of obs is position)
                pos = float(obs[0])
                max_pos_reached = max(max_pos_reached, pos)
                min_pos_reached = min(min_pos_reached, pos)
                if (step % 10) == 0:
                    trajectory.append(obs[:2])  # first 2 dims
                if terminated or truncated:
                    break

        avg_reward = total_reward / self.n_episodes
        avg_entropy = total_entropy / max(total_steps, 1)
        # Behavior signature: [max_pos, min_pos, trajectory flattened]
        traj_arr = np.array(trajectory).flatten() if trajectory else np.zeros(2)
        # Pad/truncate to fixed length
        max_len = 20
        if len(traj_arr) < max_len:
            traj_arr = np.concatenate([traj_arr, np.zeros(max_len - len(traj_arr))])
        else:
            traj_arr = traj_arr[:max_len]
        behavior = np.concatenate([[max_pos_reached, min_pos_reached], traj_arr])
        return avg_reward, avg_entropy, behavior

    def step(self, fitness_fn):
        # fitness_fn is ignored - v9 does its own eval
        T = self.train_temp()

        # 1. Sample from all species
        all_samples = []
        for k, sp in enumerate(self.species):
            samples = sp.sample()
            for s in samples:
                all_samples.append(s)
        all_samples = np.array(all_samples)

        # 2. Evaluate with stochastic policy
        rewards = np.zeros(len(all_samples))
        entropies = np.zeros(len(all_samples))
        behaviors = np.zeros((len(all_samples), 22))  # 2 + 20
        for i, x in enumerate(all_samples):
            r, e, b = self._eval_stochastic(x, T)
            rewards[i] = r
            entropies[i] = e
            behaviors[i] = b

        # 3. Adjusted fitness = reward + entropy_weight * entropy
        adjusted = rewards + self.entropy_weight * entropies * 10  # scale entropy
        # Make positive for CMA-ES
        adjusted_pos = adjusted - adjusted.min() + 1e-3

        # Track best by raw reward
        best_idx = int(np.argmax(rewards))
        if rewards[best_idx] > self.best_fitness:
            self.best_fitness = float(rewards[best_idx])
            g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
            g.params = all_samples[best_idx].copy()
            self.best_genome = g

        # 4. Assign to species by behavior
        species_beh = np.array([
            self._species_behavior(sp.center) for sp in self.species
        ])
        dists = np.linalg.norm(behaviors[:, None, :] - species_beh[None, :, :], axis=2)
        assigned = np.argmin(dists, axis=1)

        # 5. Update species
        for k, sp in enumerate(self.species):
            mask = assigned == k
            if mask.sum() < 2:
                continue
            sp.update(all_samples[mask], adjusted_pos[mask])

            if self.l0_pressure > 0:
                gate_logits = sp.center[:self.n_conns]
                weights_vec = sp.center[self.n_conns:2 * self.n_conns]
                active_small = (gate_logits > 0) & (np.abs(weights_vec) < self.l0_threshold)
                gate_logits[active_small] -= self.l0_pressure
                sp.center[:self.n_conns] = gate_logits

        # 6. Merge / restart
        if self.generation > 0 and self.generation % 10 == 0:
            self._merge_and_restart(behaviors)

        best_fit = float(np.max(rewards))
        mean_fit = float(np.mean(rewards))
        self.generation += 1
        return best_fit, mean_fit

    def _species_behavior(self, params):
        """Quick behavior estimate for a species center (deterministic)."""
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

    def _merge_and_restart(self, behaviors):
        species_beh = np.array([self._species_behavior(sp.center) for sp in self.species])
        n = len(self.species)
        merged = set()
        for i in range(n):
            if i in merged:
                continue
            for j in range(i + 1, n):
                if j in merged:
                    continue
                d = np.linalg.norm(species_beh[i] - species_beh[j])
                if d < self.merge_threshold:
                    merged.add(j)
                    break
        if merged:
            keep = [i for i in range(n) if i not in merged]
            self.species = [self.species[i] for i in keep]

        for i, sp in enumerate(self.species):
            if sp.stagnation > self.stagnation_limit:
                if self.best_genome is not None:
                    new_center = self.best_genome.params + np.random.randn(self.dim) * 0.5
                else:
                    new_center = np.random.randn(self.dim) * 0.5
                self.species[i] = Species(self.dim, new_center, sigma0=0.5, pop_size=sp.pop_size)

        while len(self.species) < self.n_species:
            new_center = np.random.randn(self.dim) * 0.5
            self.species.append(Species(self.dim, new_center, sigma0=0.5, pop_size=max(4, self.pop_size // self.n_species)))

    def best_genome_dict(self):
        if self.best_genome is not None:
            return self.best_genome.to_genome_dict()
        g = MorphGenomeV4(self.n_inputs, self.n_outputs, self.n_hidden_max)
        g.params = self.species[0].center
        return g.to_genome_dict()
