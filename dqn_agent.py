"""DQN (Deep Q-Network) with uniform replay and a frozen target network."""

from collections import deque
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from random import Random

import numpy as np
import torch
from torch import nn

from config import CONFIG


class QNetwork(nn.Module):
    """Map an observation to one discounted-return estimate per action."""

    def __init__(self, observation_size=8, action_count=3, hidden_size=64):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation):
        return self.layers(observation)


class DQNAgent:
    """Train action values with replay, a Smooth L1 loss, and hard target copies."""

    def __init__(self, seed=7, settings=None):
        self.settings = deepcopy(CONFIG)
        if settings is not None:
            self.settings.update(deepcopy(settings))
        self.seed = int(seed)
        self.metadata = {}
        for name in (
            "observation_size", "action_count", "hidden_size",
            "replay_capacity", "batch_size", "train_every",
            "target_update_steps", "torch_threads",
        ):
            value = self.settings[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.settings["warmup_steps"] < 0:
            raise ValueError("warmup_steps cannot be negative.")
        if self.settings["replay_capacity"] < max(
            self.settings["batch_size"], self.settings["warmup_steps"]
        ):
            raise ValueError("Replay capacity must cover batch size and warmup.")
        if not 0.0 <= self.settings["gamma"] <= 1.0:
            raise ValueError("gamma must be between zero and one.")
        if self.settings["learning_rate"] <= 0 or self.settings["gradient_clip_norm"] <= 0:
            raise ValueError("Learning rate and gradient limit must be positive.")

        torch.set_num_threads(self.settings["torch_threads"])
        torch.manual_seed(self.seed)
        # Separate streams keep exploration choices independent of replay sampling.
        self.action_rng = Random(self.seed)
        self.replay_rng = Random(self.seed + 1)
        self.network = QNetwork(
            self.settings["observation_size"],
            self.settings["action_count"],
            self.settings["hidden_size"],
        ).cpu()
        self.target_network = deepcopy(self.network)
        self.target_network.requires_grad_(False)
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=self.settings["learning_rate"]
        )
        self.loss_function = nn.SmoothL1Loss()
        self.memory = deque(maxlen=self.settings["replay_capacity"])
        self.environment_steps = 0
        self.optimization_steps = 0
        self._last_learning_step = -1

    def _observation_array(self, observation):
        array = np.asarray(observation, dtype=np.float32)
        if array.shape != (self.settings["observation_size"],):
            raise ValueError("Observation has the wrong number of features.")
        if not np.isfinite(array).all():
            raise ValueError("Observation must contain finite values.")
        return array

    def select_action(self, obs, epsilon=0.0):
        """Explore with probability epsilon; otherwise choose the highest value."""
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between zero and one.")
        array = self._observation_array(obs)
        # Evaluation does not consume exploration randomness.
        if epsilon > 0.0 and self.action_rng.random() < epsilon:
            return self.action_rng.randrange(self.settings["action_count"])
        self.network.eval()
        with torch.no_grad():
            values = self.network(torch.from_numpy(array).unsqueeze(0))
        return int(values.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        """Store one copied transition and advance the environment-step counter."""
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.settings["action_count"]:
            raise ValueError("Action index is outside the available actions.")
        reward = float(reward)
        if not isfinite(reward):
            raise ValueError("Reward must be finite.")
        # Copy arrays so later changes cannot rewrite remembered experience.
        self.memory.append((
            self._observation_array(state).copy(), int(action), reward,
            self._observation_array(next_state).copy(), bool(done),
        ))
        self.environment_steps += 1

    def learn(self):
        """Update once per eligible transition count, or return None when skipped.

        Warmup requires enough stored transitions. The training cadence counts
        transitions, while target synchronization counts completed optimizer steps.
        """
        if len(self.memory) < max(
            self.settings["batch_size"], self.settings["warmup_steps"]
        ):
            return None
        if self.environment_steps % self.settings["train_every"] != 0:
            return None
        # Repeated calls without new experience must not duplicate an update.
        if self.environment_steps == self._last_learning_step:
            return None

        batch = self.replay_rng.sample(list(self.memory), self.settings["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.from_numpy(np.stack(states))
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.from_numpy(np.stack(next_states))
        dones = torch.tensor(dones, dtype=torch.bool)

        self.network.train()
        predicted_values = self.network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            # Standard DQN: the frozen network both selects and values the next action.
            next_values = self.target_network(next_states).max(dim=1).values
            # A completed finite episode has no future reward to bootstrap.
            next_values = next_values.masked_fill(dones, 0.0)
            targets = rewards + self.settings["gamma"] * next_values
        loss = self.loss_function(predicted_values, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; inspect experiment settings.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.network.parameters(), self.settings["gradient_clip_norm"]
        )
        self.optimizer.step()
        self.optimization_steps += 1
        self._last_learning_step = self.environment_steps
        if self.optimization_steps % self.settings["target_update_steps"] == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        return float(loss.item())

    def save(self, path, metadata=None):
        """Save inference weights, settings, seed, and serializable metadata.

        Metadata must support JSON (JavaScript Object Notation) serialization.

        A sibling temporary file is written before replacing the destination.
        Replay, optimizer state, and training counters are intentionally omitted.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata = json.loads(json.dumps(
            self.metadata if metadata is None else metadata
        ))
        checkpoint = {
            "model_state_dict": self.network.state_dict(),
            "settings": json.loads(json.dumps(self.settings)),
            "metadata": self.metadata,
            "seed": self.seed,
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)

    @classmethod
    def load(cls, path):
        """Load for evaluation; optimizer and replay history are not resumed."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        agent = cls(seed=checkpoint.get("seed", 7), settings=checkpoint["settings"])
        agent.network.load_state_dict(checkpoint["model_state_dict"])
        agent.network.eval()
        agent.target_network.load_state_dict(agent.network.state_dict())
        agent.target_network.eval()
        agent.metadata = checkpoint.get("metadata", {})
        return agent


if __name__ == "__main__":
    from rl_environment import LearningEnvironment

    torch.manual_seed(42)

    example_task = {
        "data_mbits": 1.0,
        "work_gcycles": 1.0,
        "deadline_s": 0.4,
        "uplink_mbps": 20.0,
    }

    environment = LearningEnvironment(
        [example_task, example_task]
    )
    observation = environment.reset()

    network = QNetwork()
    network.eval()

    observation_tensor = torch.as_tensor(
        observation, dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = network(observation_tensor)

    action = int(q_values.argmax(dim=1).item())

    print("Input shape:", tuple(observation_tensor.shape))
    print("Output shape:", tuple(q_values.shape))
    print("\nUntrained action-value estimates:")

    for index, location in enumerate(environment.ACTIONS):
        print(f"  {location}: {q_values[0, index].item():.4f}")

    print(
        "\nAction chosen by the untrained network:",
        environment.ACTIONS[action]
    )

    next_observation, reward, done, info = environment.step(action)

    print(f"Actual simulated delay: {info['delay_s']:.2f} seconds")
    print(f"Deadline met: {info['deadline_met']}")
    print(f"Actual reward: {reward:.2f}")
