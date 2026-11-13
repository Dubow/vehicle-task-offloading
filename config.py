"""Shared settings for reproducible task-offloading training and evaluation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

CONFIG = {
    # Feature count and action order must match LearningEnvironment.
    "observation_size": 8,
    "action_count": 3,
    "hidden_size": 64,  # Units in each of the two hidden layers.
    "learning_rate": 0.001,
    "gamma": 0.99,  # Discount per task decision.
    # Replay sizes and warmup are measured in stored task transitions.
    "replay_capacity": 20_000,
    "batch_size": 64,
    "warmup_steps": 1_000,
    "train_every": 4,  # Attempt one update every this many task transitions.
    "target_update_steps": 200,  # Copy weights after this many optimizer updates.
    "gradient_clip_norm": 10.0,  # Maximum combined Euclidean gradient norm.
    "torch_threads": 1,
    # Each episode starts with empty server queues and a newly seeded task trace.
    "episodes": 500,
    "tasks_per_episode": 100,
    "arrival_gap_s": 0.2,  # Seconds between consecutive task arrivals.
    "seeds": [7, 17, 27],
    # Trace seed = agent seed * stride + one-based episode number.
    "training_seed_stride": 10_000,
    # Epsilon is a random-action probability; decay uses a fraction of episodes.
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_fraction": 0.8,
    "validation_interval": 25,  # Episodes between checks; also check the final one.
    # Validation selects checkpoints; test seeds are reserved for final evaluation.
    "validation_seeds": list(range(1_000_000, 1_000_010)),
    "test_seed_start": 2_000_000,
}
