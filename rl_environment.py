from math import log1p

import numpy as np

from environment import OffloadingEnvironment


class LearningEnvironment:
    ACTIONS = ("Local", "Edge", "Cloud")

    def __init__(self, tasks, arrival_gap_s=0.2):
        if not tasks:
            raise ValueError("Provide at least one task.")

        self.tasks = [task.copy() for task in tasks]
        self.simulator = OffloadingEnvironment(
            arrival_gap_s=arrival_gap_s
        )
        self.index = 0

    def reset(self):
        self.simulator.reset()
        self.index = 0
        return self.observation()

    def observation(self):
        if self.index >= len(self.tasks):
            return np.zeros(8, dtype=np.float32)

        task = self.tasks[self.index]

        reserved_times = [
            max(
                0.0,
                self.simulator.server_free_at[name]
                - self.simulator.time_s
            )
            for name in self.ACTIONS
        ]

        return np.array([
            task["data_mbits"] / 1.5,
            task["work_gcycles"] / 3.0,
            task["deadline_s"] / 1.2,
            task["uplink_mbps"] / 40.0,
            log1p(reserved_times[0]),
            log1p(reserved_times[1]),
            log1p(reserved_times[2]),
            (len(self.tasks) - self.index) / len(self.tasks),
        ], dtype=np.float32)

    def step(self, action):
        if self.index >= len(self.tasks):
            raise RuntimeError("Episode finished. Call reset() first.")

        if action not in (0, 1, 2):
            raise ValueError("Action must be 0, 1, or 2.")

        task = self.tasks[self.index]
        location = self.ACTIONS[action]

        self.simulator.uplink_mbps = task["uplink_mbps"]

        result = self.simulator.step(
            location,
            task["data_mbits"],
            task["work_gcycles"]
        )

        delay = result["delay_s"]
        deadline_met = delay <= task["deadline_s"] + 1e-9

        reward = -(delay / 1.0) - (
            0.0 if deadline_met else 1.0
        )

        self.index += 1
        done = self.index == len(self.tasks)

        info = {
            **result,
            "action": location,
            "deadline_s": task["deadline_s"],
            "deadline_met": deadline_met,
        }

        return self.observation(), reward, done, info


if __name__ == "__main__":
    example_task = {
        "data_mbits": 1.0,
        "work_gcycles": 1.0,
        "deadline_s": 0.4,
        "uplink_mbps": 20.0,
    }

    environment = LearningEnvironment(
        [example_task, example_task]
    )

    np.set_printoptions(precision=3, suppress=True)

    print("Initial observation:", environment.reset())

    # First task: Edge. Second task: Local.
    for action in (1, 0):
        observation, reward, done, info = environment.step(action)

        print(f"\nAction: {info['action']}")
        print(f"Delay: {info['delay_s']:.2f} seconds")
        print(f"Deadline met: {info['deadline_met']}")
        print(f"Reward: {reward:.2f}")
        print("Next observation:", observation)
        print("Episode finished:", done)