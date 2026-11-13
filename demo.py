"""Demonstrate a saved DQN (Deep Q-Network) without retraining it."""

import argparse
from pathlib import Path
from statistics import mean

import torch

from baselines import generate_tasks
from dqn_agent import DQNAgent
from rl_environment import LearningEnvironment


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=3_000_000)
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("--tasks must be positive")

    if args.model is None:
        checkpoints = sorted((ROOT / "models").glob("dqn_seed_*.pt"))
        if not checkpoints:
            parser.error("No trained models found. Run train.py first.")
        args.model = checkpoints[0]

    torch.set_num_threads(1)
    agent = DQNAgent.load(args.model)
    tasks = generate_tasks(args.tasks, seed=args.seed)
    environment = LearningEnvironment(tasks, arrival_gap_s=0.2)
    observation = environment.reset()
    delays, successes = [], 0

    print("Vehicle task offloading using DQN (Deep Q-Network)")
    print(f"Saved model: {args.model.name}")
    print(f"Demonstration seed: {args.seed}; {args.tasks} tasks; no training or exploration")
    print("Default model selection uses filename order, not final-test performance.\n")
    print("Task | Input Mb | Work Gcycles | Link Mbps | Deadline s | Choice | Delay s | Met? | Reward")

    for task_id, task in enumerate(tasks, start=1):
        action = agent.select_action(observation, epsilon=0.0)
        observation, reward, done, info = environment.step(action)
        delays.append(info["delay_s"])
        successes += int(info["deadline_met"])
        outcome = "Yes" if info["deadline_met"] else "No"
        print(
            f"{task_id:>4} | {task['data_mbits']:>8.2f} | "
            f"{task['work_gcycles']:>12.2f} | {task['uplink_mbps']:>9.1f} | "
            f"{task['deadline_s']:>10.2f} | {info['action']:<6} | "
            f"{info['delay_s']:>7.3f} | {outcome:<4} | {reward:>6.3f}"
        )

    print(f"\nAverage delay: {mean(delays):.3f} seconds")
    print(f"Deadlines met: {successes}/{len(tasks)} ({100 * successes / len(tasks):.1f}%)")
    print("Mb means megabits; Gcycles means billions of processor cycles; Mbps means megabits per second.")
    print("This short demonstration is separate from the larger final evaluation.")


if __name__ == "__main__":
    main()
