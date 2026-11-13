# Experiment results

These outputs describe a synthetic task-offloading simulation. Delay is
simulated time from task arrival to completion, not the wall-clock time needed
to run Python. Each policy receives the same 30 test sequences of 100 tasks,
starting with empty processing queues. Its own choices determine later queues.

| Artifact | How to read it |
|---|---|
| `summary.csv` | One row per baseline or trained model. Mean delay is in seconds; deadline success and destination shares are percentages. |
| `evaluation_results.csv` | One row per policy and test episode, including task counts, mean reward, delay, deadline success, and destination counts. |
| `test_scenarios.csv` | The 3,000 shared task inputs with task identifiers, arrival times, data sizes, required work, deadlines, and upload rates. |
| `training_seed_*.csv` | Training progress by episode, including exploration probability, loss, and validation metrics at validation checkpoints. |
| `evaluation_config.json` | Test seed range, scenario settings, checkpoint hashes, and evaluation rules. |
| `implementation_snapshot.json` | Source hashes recorded when the experiment was produced, plus a summary of the seven implementation checks. |

CSV (Comma-Separated Values) tables can be opened in a spreadsheet application
or read in Python. In `summary.csv`, standard deviations describe variation
across the 30 episode means for each policy or model. They are not confidence
intervals and do not summarize variation across the three training seeds.

## Figures

- [`average_delay.png`](figures/average_delay.png): lower is better. The vertical
  axis is logarithmic because the overloaded local policy has much larger delays.
- [`deadline_success.png`](figures/deadline_success.png): higher is better. Every
  task is included, including tasks that miss their deadlines.
- [`offloading_choices.png`](figures/offloading_choices.png): the share of tasks
  assigned to each destination; each policy's shares sum to 100 percent.
- [`training_reward.png`](figures/training_reward.png): a trailing mean over up
  to 20 training episodes. Higher (less negative) reward is better. Exploration
  is active during training, so this curve is not the final evaluation score.

## Interpreting the comparison

Greedy achieves about 0.354 seconds mean delay and 88.4 percent deadline success.
The three DQN (Deep Q-Network) runs achieve about 0.331 to 0.332 seconds and
90.5 to 90.9 percent. These are descriptive results for the documented workload;
they do not establish general superiority across other system settings.

Validation seeds select checkpoints; separate test seeds assess those frozen
checkpoints. The short demonstration uses another seed and is not part of the
3,000-task benchmark. A demonstration's percentage can therefore differ from
the benchmark without indicating an error.

## Reproduction and provenance

From the project root, rerun the comparison using the supplied checkpoints:

```powershell
.\.venv\Scripts\python.exe evaluate.py --episodes 30 --output-dir results/recheck
```

This preserves the original outputs. A different output directory needs its
own training logs to generate the training-reward figure; the comparison
tables and three policy figures use the saved models directly.

The original implementation and artifacts were committed in `e6a08c4`.
Subsequent comments and documentation can change source-file hashes while
leaving executable behavior unchanged. The historical implementation snapshot
is intentionally retained rather than relabeled as a new experiment.
