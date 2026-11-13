"""Calculate transfer, waiting, and processing delays on three destinations.

This is a numerical simulation: it reserves future processing slots instead of
sleeping or sending data to real servers. Each call dispatches one whole task.
"""


class OffloadingEnvironment:
    def __init__(self, uplink_mbps=20.0, arrival_gap_s=0.1):
        if uplink_mbps <= 0 or arrival_gap_s <= 0:
            raise ValueError("Network speed and arrival gap must be positive.")

        self.uplink_mbps = uplink_mbps
        self.arrival_gap_s = arrival_gap_s

        # Computing speeds: billions of CPU (Central Processing Unit) cycles/s.
        self.cpu_rates = {
            "Local": 1.0,
            "Edge": 5.0,
            "Cloud": 10.0,
        }

        # Additional communication delay, in seconds.
        self.extra_network_s = {
            "Local": 0.0,
            "Edge": 0.0,
            "Cloud": 0.3,
        }

        self.reset()

    def reset(self):
        """Return the simulated clock and all destination reservations to zero."""
        self.time_s = 0.0
        self.server_free_at = {
            name: 0.0 for name in self.cpu_rates
        }

    def step(self, action, data_mbits=1.0, work_gcycles=1.0):
        """Reserve processing in dispatch order and return delays in seconds.

        Input data is measured in megabits and work in billions of cycles.
        Queue time starts after transfer; total delay includes transfer, queue
        time, and computation. The clock advances by the arrival interval, not
        by completion time, so later tasks can arrive while work is pending.
        """
        if action not in self.cpu_rates:
            raise ValueError("Choose Local, Edge, or Cloud.")

        if data_mbits <= 0 or work_gcycles <= 0:
            raise ValueError("Task size and required work must be positive.")

        # Time when the vehicle generates this task.
        arrival_s = self.time_s

        upload_s = (
            0.0 if action == "Local"
            else data_mbits / self.uplink_mbps
        )

        transfer_s = upload_s + self.extra_network_s[action]
        ready_s = arrival_s + transfer_s

        # Both the task and the server must be ready.
        start_s = max(ready_s, self.server_free_at[action])

        queue_s = start_s - ready_s
        compute_s = work_gcycles / self.cpu_rates[action]
        finish_s = start_s + compute_s

        # Reserve this server until the task finishes.
        self.server_free_at[action] = finish_s

        # Advance to the next task's generation time.
        self.time_s += self.arrival_gap_s

        return {
            "arrival_s": arrival_s,
            "queue_s": queue_s,
            "delay_s": finish_s - arrival_s,
        }


if __name__ == "__main__":
    environment = OffloadingEnvironment(arrival_gap_s=0.3)

    print("Task | Arrival (s) | Queue wait (s) | Total delay (s)")

    for task_id in range(1, 7):
        result = environment.step("Edge")

        print(
            f"{task_id:>4} | "
            f"{result['arrival_s']:>11.2f} | "
            f"{result['queue_s']:>14.2f} | "
            f"{result['delay_s']:>15.2f}"
        )
