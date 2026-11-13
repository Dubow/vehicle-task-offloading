"""Generate reproducible synthetic tasks and compare five reference policies.

Local, Edge, and Cloud always use their named destination. Random samples a
destination uniformly; Greedy chooses the smallest estimated current delay.
"""

from copy import deepcopy
from random import Random
from statistics import mean

from environment import OffloadingEnvironment

ACTIONS = ("Local", "Edge", "Cloud")


def generate_tasks(count=100, seed=42):
    """Return repeatable task inputs; deadlines are relative to task arrival.

    Sizes use megabits, work uses billions of cycles, and upload rates use
    megabits per second. A private random generator isolates the task sequence
    from policy exploration and other random operations elsewhere in Python.
    """
    rng = Random(seed)
    tasks = []

    for _ in range(count):
        data_mbits = rng.choice([0.5, 1.0, 1.5])

        tasks.append({
            "data_mbits": data_mbits,
            "work_gcycles": data_mbits * rng.choice([0.5, 1.0, 2.0]),
            "deadline_s": rng.choice([0.4, 0.8, 1.2]),
            "uplink_mbps": rng.choice([10.0, 20.0, 40.0]),
        })

    return tasks


def choose_action(policy, environment, task, rng):
    """Select a destination without modifying the real simulation state.

    The caller sets the current upload rate before Greedy previews each action
    on a copy. Greedy optimizes this task's delay, not future queue congestion
    or the explicit deadline penalty used by the learning agent.
    """
    if policy in ACTIONS:
        return policy

    if policy == "Random":
        return rng.choice(ACTIONS)

    if policy == "Greedy":
        predicted_delays = {}

        for action in ACTIONS:
            preview = deepcopy(environment)

            result = preview.step(
                action, task["data_mbits"], task["work_gcycles"]
            )

            predicted_delays[action] = result["delay_s"]

        return min(
            predicted_delays,
            key=lambda action: predicted_delays[action]
        )

    raise ValueError(f"Unknown policy: {policy}")


def evaluate_policy(policy, tasks):
    """Return mean delay, deadline-success percentage, and destination counts.

    Every policy starts with empty queues and processes the provided inputs in
    order. The standalone seed-42 example is separate from final evaluation.
    """
    environment = OffloadingEnvironment(arrival_gap_s=0.2)
    rng = Random(7)

    delays = []
    deadlines_met = 0
    choices = {action: 0 for action in ACTIONS}

    for task in tasks:
        environment.uplink_mbps = task["uplink_mbps"]

        action = choose_action(policy, environment, task, rng)

        result = environment.step(
            action, task["data_mbits"], task["work_gcycles"]
        )

        delay = result["delay_s"]

        delays.append(delay)
        deadlines_met += delay <= task["deadline_s"] + 1e-9
        choices[action] += 1

    return mean(delays), 100 * deadlines_met / len(tasks), choices


if __name__ == "__main__":
    tasks = generate_tasks()

    print("100 shared tasks; one task arrives every 0.2 seconds")
    print("Policy   | Mean delay (s) | Deadlines met | Local Edge Cloud")

    for policy in (*ACTIONS, "Random", "Greedy"):
        average, success, counts = evaluate_policy(policy, tasks)

        print(
            f"{policy:<8} | {average:>14.3f} | {success:>12.1f}% | "
            f"{counts['Local']:>5} {counts['Edge']:>4} {counts['Cloud']:>5}"
        )
