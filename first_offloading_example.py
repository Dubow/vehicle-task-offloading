"""Illustrate how upload speed changes a single task's fastest destination.

This introductory calculation has no queues or learning. Its fixed task and
network overheads differ from the later multi-task experiment settings.
"""


def estimate_delays(uplink_mbps):
    """Return delays in seconds for an upload rate in megabits per second."""
    task_mbits = 8.0       # 1 MB (megabyte) = 8 megabits of input data.
    task_gcycles = 1.0     # 1 billion CPU (Central Processing Unit) cycles.

    upload_seconds = task_mbits / uplink_mbps

    return {
        "Local": task_gcycles / 1.0,
        "Edge": upload_seconds + 0.05 + task_gcycles / 5.0,
        "Cloud": upload_seconds + 0.30 + 0.02 + task_gcycles / 10.0,
    }


for rate in (20, 2):
    delays = estimate_delays(rate)
    best = min(delays, key=delays.get)

    print(f"\nUplink: {rate} Mbps")

    for location, seconds in delays.items():
        print(f"  {location}: {seconds:.2f} seconds")

    print(f"  Lowest-delay choice: {best}")
