# Saved models

These checkpoints contain trained DQN (Deep Q-Network) policies. Each network
maps eight observation values to three action-value estimates, in the order
Local, Edge, and Cloud. A random seed controls initialization and sampling;
it is an experiment identifier, not a quality ranking.

| Files | Purpose |
|---|---|
| `dqn_seed_7.pt`, `dqn_seed_17.pt`, `dqn_seed_27.pt` | Learned network weights, settings, seed, and checkpoint-selection metadata. |
| `config_seed_7.json`, `config_seed_17.json`, `config_seed_27.json` | Recorded hyperparameters, task-seed streams, software versions, completed training budget, and validation outcome for the selected checkpoint. |

Each run completed 500 episodes of 100 tasks. The saved checkpoint is the one
with the highest mean validation episode reward, with the earlier checkpoint
retained on ties. It need not be the last episode. All three checkpoints are
evaluated on the final test set; test results do not choose the checkpoints.

From the project root, run a specific model:

```powershell
.\.venv\Scripts\python.exe demo.py --model models/dqn_seed_7.pt --tasks 10
```

`DQNAgent.load()` restores the learned network for inference on the CPU (Central
Processing Unit). The checkpoint does not restore replay memory or optimizer
state for exact training continuation. Run `train.py` for a new training run.

The `.pt` files are binary model artifacts. Preserve them as generated; the
SHA-256 (Secure Hash Algorithm, 256-bit) digests in
[`evaluation_config.json`](../results/evaluation_config.json) identify the exact
checkpoints used in the recorded comparison.
