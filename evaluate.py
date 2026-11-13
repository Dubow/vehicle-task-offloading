"""Compare frozen DQN (Deep Q-Network) models with five baseline policies."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from random import Random
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from baselines import ACTIONS, choose_action, generate_tasks
from dqn_agent import DQNAgent
from rl_environment import LearningEnvironment


ROOT = Path(__file__).resolve().parent
TASK_COUNT = 100
ARRIVAL_GAP_S = 0.2
TEST_SEED_START = 2_000_000


def write_csv(path, rows):
    if not rows:
        raise ValueError("There are no rows to save.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_episode(policy, tasks, test_seed, agent=None):
    """Replay external inputs; each policy creates its own queues."""
    environment = LearningEnvironment(tasks, arrival_gap_s=ARRIVAL_GAP_S)
    observation = environment.reset()
    rng = Random(7 + test_seed)
    delays, rewards = [], []
    counts = {name: 0 for name in ACTIONS}
    successes = 0

    for task in tasks:
        if agent is None:
            # Greedy previews must see this task's current network rate.
            environment.simulator.uplink_mbps = task["uplink_mbps"]
            location = choose_action(policy, environment.simulator, task, rng)
            action = ACTIONS.index(location)
        else:
            action = agent.select_action(observation, epsilon=0.0)

        observation, reward, done, info = environment.step(action)
        delays.append(info["delay_s"])
        rewards.append(reward)
        successes += int(info["deadline_met"])
        counts[info["action"]] += 1

    if not done or sum(counts.values()) != len(tasks):
        raise RuntimeError("Evaluation did not account for every task.")

    return {
        "task_count": len(tasks),
        "mean_delay_s": mean(delays),
        "deadline_success_pct": 100.0 * successes / len(tasks),
        "mean_reward": mean(rewards),
        "local_count": counts["Local"],
        "edge_count": counts["Edge"],
        "cloud_count": counts["Cloud"],
    }


def summarize(rows):
    summaries = []
    for policy in dict.fromkeys(row["policy"] for row in rows):
        selected = [row for row in rows if row["policy"] == policy]
        summary = {
            "policy": policy,
            "training_seed": selected[0]["training_seed"],
            "episodes": len(selected),
        }
        for source, average_name, spread_name in (
            ("mean_delay_s", "mean_delay_s", "std_delay_s"),
            ("deadline_success_pct", "mean_deadline_success_pct",
             "std_deadline_success_pct"),
            ("mean_reward", "mean_reward", "std_mean_reward"),
        ):
            values = [row[source] for row in selected]
            summary[average_name] = mean(values)
            summary[spread_name] = stdev(values) if len(values) > 1 else 0.0
        total_tasks = sum(row["task_count"] for row in selected)
        for destination in ("local", "edge", "cloud"):
            summary[f"{destination}_pct"] = (
                100 * sum(row[f"{destination}_count"] for row in selected)
                / total_tasks
            )
        summaries.append(summary)
    return summaries


def make_figures(summaries, output_dir):
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["policy"] for row in summaries]
    positions = list(range(len(labels)))
    colors = ["#0c8599" if label.startswith("DQN") else "#8793a5"
              for label in labels]

    for key, filename, ylabel, title, percent in (
        ("mean_delay_s", "average_delay.png",
         "Mean task completion delay (seconds; logarithmic scale)",
         "Completion delay on unseen task sequences", False),
        ("mean_deadline_success_pct", "deadline_success.png",
         "Tasks meeting their deadline (%)",
         "Deadline success on unseen task sequences", True),
    ):
        fig, axis = plt.subplots(figsize=(11, 5.5))
        values = [row[key] for row in summaries]
        axis.bar(positions, values, color=colors)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.set_title(title + "\nDQN (Deep Q-Network); arrivals every 0.2 seconds")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        if percent:
            axis.set_ylim(0, 108)
        else:
            axis.set_yscale("log")
            axis.set_ylim(bottom=min(values) * 0.65, top=max(values) * 2)
        for position, value in zip(positions, values):
            label = f"{value:.1f}%" if percent else f"{value:.3f}"
            axis.annotate(label, (position, value), xytext=(0, 5),
                          textcoords="offset points", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5.5))
    bottom = [0.0] * len(summaries)
    for destination, color in (("Local", "#4263eb"), ("Edge", "#12b886"),
                               ("Cloud", "#f59f00")):
        values = [row[f"{destination.lower()}_pct"] for row in summaries]
        axis.bar(positions, values, bottom=bottom, label=destination, color=color)
        bottom = [a + b for a, b in zip(bottom, values)]
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Share of assigned tasks (%)")
    axis.set_title("Processing destinations\nDQN (Deep Q-Network); arrivals every 0.2 seconds")
    axis.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(figures / "offloading_choices.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)

    # Training reward is diagnostic; final comparisons use the separate test data.
    logs = sorted(output_dir.glob("training_seed_*.csv"))
    if logs:
        fig, axis = plt.subplots(figsize=(10, 5))
        for path in logs:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                entries = list(csv.DictReader(stream))
            if not entries:
                continue
            reward_key = next((key for key in ("reward", "episode_reward", "total_reward")
                               if key in entries[0]), None)
            if reward_key is None:
                continue
            episode_values = [int(row["episode"]) for row in entries]
            rewards = [float(row[reward_key]) for row in entries]
            smoothed = [mean(rewards[max(0, index - 19):index + 1])
                        for index in range(len(rewards))]
            seed = path.stem.removeprefix("training_seed_")
            axis.plot(episode_values, smoothed, label=f"Training seed {seed}")
        axis.set_xlabel("Training episode")
        axis.set_ylabel("Episode reward (trailing mean, up to 20 episodes)")
        axis.set_title("DQN (Deep Q-Network) training progress")
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend()
        fig.tight_layout()
        fig.savefig(figures / "training_reward.png", dpi=180, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    checkpoints = sorted(args.model_dir.glob("dqn_seed_*.pt"))
    if not checkpoints:
        parser.error("No trained models found. Run train.py first.")

    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = [(TEST_SEED_START + index,
                  generate_tasks(TASK_COUNT, seed=TEST_SEED_START + index))
                 for index in range(args.episodes)]
    scenario_rows = []
    for episode, (test_seed, tasks) in enumerate(scenarios, start=1):
        for task_index, task in enumerate(tasks):
            scenario_rows.append({"episode": episode, "test_seed": test_seed,
                                  "task_id": task_index + 1,
                                  "arrival_time_s": task_index * ARRIVAL_GAP_S,
                                  **task})
    write_csv(args.output_dir / "test_scenarios.csv", scenario_rows)

    policies = [(name, "", None) for name in (*ACTIONS, "Random", "Greedy")]
    for path in checkpoints:
        seed = int(path.stem.removeprefix("dqn_seed_"))
        agent = DQNAgent.load(path)
        if (agent.settings["tasks_per_episode"] != TASK_COUNT
                or agent.settings["arrival_gap_s"] != ARRIVAL_GAP_S):
            parser.error(f"{path.name}: training and evaluation scenario settings differ")
        policies.append((f"DQN seed {seed}", seed, agent))

    rows = []
    for policy, training_seed, agent in policies:
        for episode, (test_seed, tasks) in enumerate(scenarios, start=1):
            metrics = evaluate_episode(policy, tasks, test_seed, agent)
            rows.append({"policy": policy, "training_seed": training_seed,
                         "episode": episode, "test_seed": test_seed, **metrics})
        print(f"Evaluated {policy}: {args.episodes} shared test episodes", flush=True)

    summaries = summarize(rows)
    write_csv(args.output_dir / "evaluation_results.csv", rows)
    write_csv(args.output_dir / "summary.csv", summaries)
    metadata = {
        "task_count_per_episode": TASK_COUNT,
        "arrival_gap_s": ARRIVAL_GAP_S,
        "test_seed_start": TEST_SEED_START,
        "test_seed_end_inclusive": TEST_SEED_START + args.episodes - 1,
        "episode_count": args.episodes,
        "random_policy_seed": "7 + test_seed",
        "exploration_probability": 0.0,
        "learning_during_evaluation": False,
        "checkpoint_selection": "All supplied checkpoints; no test-based selection",
        "checkpoints": [{"filename": path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for path in checkpoints],
        "variability": "Sample standard deviation across episode means, separately for each policy/model. Not a confidence interval.",
        "scope": "Synthetic tasks; fixed processing capacities; dispatch-order reservations; no separate upload queue or return-data delay.",
    }
    (args.output_dir / "evaluation_config.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_figures(summaries, args.output_dir)

    print("\nDQN (Deep Q-Network): frozen-model evaluation")
    print(f"{'Policy':<15} {'Mean delay (s)':>15} {'Deadlines met':>15}")
    for row in summaries:
        print(f"{row['policy']:<15} {row['mean_delay_s']:>15.3f} "
              f"{row['mean_deadline_success_pct']:>14.1f}%")
    print(f"\nSaved tables, settings, test scenarios, and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
