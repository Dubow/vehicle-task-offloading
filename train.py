"""Train DQN [Deep Q-Network] models, selected using validation traces."""

import argparse
from copy import deepcopy
import csv
import json
from math import ceil
from pathlib import Path
import platform
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from baselines import generate_tasks
from config import CONFIG, ROOT
from dqn_agent import DQNAgent
from rl_environment import LearningEnvironment


def validate(agent, settings):
    """Measure frozen behavior without exploration, replay, or updates."""
    returns, delays, deadlines_met = [], [], 0
    for task_seed in settings["validation_seeds"]:
        environment = LearningEnvironment(
            generate_tasks(count=settings["tasks_per_episode"], seed=task_seed),
            arrival_gap_s=settings["arrival_gap_s"],
        )
        observation = environment.reset()
        episode_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, done, info = environment.step(action)
            episode_reward += reward
            delays.append(info["delay_s"])
            deadlines_met += int(info["deadline_met"])
        returns.append(episode_reward)
    return {
        "reward": mean(returns),
        "mean_delay_s": mean(delays),
        "deadline_success_pct": 100.0 * deadlines_met / len(delays),
    }


def exploration_probability(episode, settings):
    intervals = max(1, ceil(settings["episodes"] * settings["epsilon_decay_fraction"]) - 1)
    progress = min(1.0, (episode - 1) / intervals)
    return settings["epsilon_start"] + progress * (settings["epsilon_end"] - settings["epsilon_start"])


def train_seed(seed, settings, output_dir):
    models_dir, results_dir = output_dir / "models", output_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"dqn_seed_{seed}.pt"
    config_path = models_dir / f"config_seed_{seed}.json"
    log_path = results_dir / f"training_seed_{seed}.csv"
    provenance = {
        "settings": deepcopy(settings), "seed": seed,
        "training_task_seeds": {
            "formula": "seed * training_seed_stride + episode (episode starts at 1)",
            "first": seed * settings["training_seed_stride"] + 1,
            "last": seed * settings["training_seed_stride"] + settings["episodes"],
        },
        "validation_task_seeds": settings["validation_seeds"],
        "reserved_test_seed_start": settings["test_seed_start"],
        "versions": {"python": platform.python_version(), "numpy": str(np.__version__),
                     "torch": str(torch.__version__), "device": "cpu"},
        "selection_rule": "Highest mean validation episode reward; first checkpoint wins ties.",
        "completed_episodes": 0,
    }
    config_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    agent = DQNAgent(seed=seed, settings=settings)
    best_validation_reward, best_metadata = float("-inf"), None
    started = perf_counter()
    fields = ["episode", "reward", "mean_delay_s", "deadline_success_pct", "epsilon", "loss",
              "validation_reward", "validation_mean_delay_s", "validation_deadline_success_pct"]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for episode in range(1, settings["episodes"] + 1):
            task_seed = seed * settings["training_seed_stride"] + episode
            environment = LearningEnvironment(
                generate_tasks(count=settings["tasks_per_episode"], seed=task_seed),
                arrival_gap_s=settings["arrival_gap_s"],
            )
            observation = environment.reset()
            epsilon = exploration_probability(episode, settings)
            episode_reward, deadlines_met = 0.0, 0
            delays, losses = [], []
            done = False
            while not done:
                action = agent.select_action(observation, epsilon=epsilon)
                next_observation, reward, done, info = environment.step(action)
                agent.remember(observation, action, reward, next_observation, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)
                observation = next_observation
                episode_reward += reward
                delays.append(info["delay_s"])
                deadlines_met += int(info["deadline_met"])
            row = {
                "episode": episode, "reward": episode_reward, "mean_delay_s": mean(delays),
                "deadline_success_pct": 100.0 * deadlines_met / len(delays), "epsilon": epsilon,
                "loss": mean(losses) if losses else "", "validation_reward": "",
                "validation_mean_delay_s": "", "validation_deadline_success_pct": "",
            }
            should_validate = episode % settings["validation_interval"] == 0 or episode == settings["episodes"]
            if should_validate:
                validation = validate(agent, settings)
                row.update({"validation_reward": validation["reward"],
                            "validation_mean_delay_s": validation["mean_delay_s"],
                            "validation_deadline_success_pct": validation["deadline_success_pct"]})
                if validation["reward"] > best_validation_reward:
                    best_validation_reward = validation["reward"]
                    best_metadata = {
                        "seed": seed, "selected_episode": episode,
                        "training_episode_budget": settings["episodes"],
                        "validation_mean_reward": validation["reward"],
                        "validation_mean_delay_s": validation["mean_delay_s"],
                        "validation_deadline_success_pct": validation["deadline_success_pct"],
                        "selection_rule": provenance["selection_rule"],
                    }
                    agent.save(model_path, metadata=best_metadata)
                provenance.update({"completed_episodes": episode,
                                   "selected_checkpoint": best_metadata,
                                   "elapsed_seconds": perf_counter() - started})
                config_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
                print(
                    f"Seed {seed} | episode {episode}/{settings['episodes']} | exploration {epsilon:.3f} | "
                    f"validation reward {validation['reward']:.2f} | "
                    f"validation delay {validation['mean_delay_s']:.3f} s | "
                    f"deadlines {validation['deadline_success_pct']:.1f}%", flush=True,
                )
            writer.writerow(row)
            if should_validate:
                handle.flush()
    print(f"Saved seed {seed}: selected episode {best_metadata['selected_episode']}; model {model_path}", flush=True)
    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=CONFIG["episodes"])
    parser.add_argument("--seeds", nargs="+", type=int, default=CONFIG["seeds"])
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.episodes <= 0 or args.episodes > CONFIG["training_seed_stride"]:
        parser.error("episodes must be between 1 and 10000 to preserve separate training traces.")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be distinct non-negative integers.")
    for seed in args.seeds:
        first = seed * CONFIG["training_seed_stride"] + 1
        last = first + args.episodes - 1
        if last >= CONFIG["test_seed_start"] or any(first <= s <= last for s in CONFIG["validation_seeds"]):
            parser.error("Training trace seeds overlap reserved validation or test traces.")
    settings = deepcopy(CONFIG)
    settings.update({"episodes": args.episodes, "seeds": list(args.seeds)})
    output_dir = args.output_dir.resolve()
    print(
        "Training DQN [Deep Q-Network] on the CPU [Central Processing Unit].\n"
        "Checkpoint selection uses validation traces only; final test traces remain unused.", flush=True,
    )
    for seed in args.seeds:
        train_seed(seed, settings, output_dir)


if __name__ == "__main__":
    main()
