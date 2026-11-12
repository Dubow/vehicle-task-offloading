def estimate_delays(uplink_mbps):
    task_mbits = 8.0       # 1 MB of input data
    task_gcycles = 1.0     # 1 billion CPU cycles of work

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