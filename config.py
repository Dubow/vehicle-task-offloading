"""Shared settings for the educational task-offloading experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONFIG = {
    "observation_size": 8,
    "action_count": 3,
    "hidden_size": 64,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "replay_capacity": 20_000,
    "batch_size": 64,
    "warmup_steps": 1_000,
    "train_every": 4,
    "target_update_steps": 200,
    "gradient_clip_norm": 10.0,
    "torch_threads": 1,
    "episodes": 500,
    "tasks_per_episode": 100,
    "arrival_gap_s": 0.2,
    "seeds": [7, 17, 27],
    "training_seed_stride": 10_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_fraction": 0.8,
    "validation_interval": 25,
    "validation_seeds": list(range(1_000_000, 1_000_010)),
    "test_seed_start": 2_000_000,
}
