"""Deterministic contracts for the simulator adapter and DQN (Deep Q-Network).

These checks establish transition and update correctness, not policy convergence
or superiority over the baselines.
"""
import copy
from pathlib import Path
from random import Random
import tempfile
import unittest

import numpy as np
import torch

from baselines import ACTIONS, choose_action, evaluate_policy, generate_tasks
from dqn_agent import DQNAgent
from rl_environment import LearningEnvironment


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(data_mbits=1.0, work_gcycles=1.0,
                         deadline_s=0.4, uplink_mbps=20.0)

    def test_observations_rewards_and_terminal(self):
        """Check queue features, delay penalties, terminal zeros, and reset behavior."""
        env = LearningEnvironment([self.task, self.task])
        initial = env.reset()
        state, reward, done, info = env.step(1)
        self.assertAlmostEqual(info['delay_s'], 0.25)
        self.assertAlmostEqual(reward, -0.25)
        self.assertFalse(done)
        self.assertAlmostEqual(float(state[5]), np.log1p(0.05), places=6)
        self.assertEqual(float(state[7]), 0.5)
        state, reward, done, info = env.step(0)
        self.assertEqual(reward, -2.0)
        self.assertTrue(done)
        np.testing.assert_array_equal(state, np.zeros(8))
        with self.assertRaises(RuntimeError):
            env.step(0)
        np.testing.assert_array_equal(env.reset(), initial)

    def test_deadline_boundary(self):
        """A task finishing exactly at its deadline receives no miss penalty."""
        task = {**self.task, 'deadline_s': 0.25}
        _, reward, _, info = LearningEnvironment([task]).step(1)
        self.assertTrue(info['deadline_met'])
        self.assertEqual(reward, -0.25)

    def test_wrapper_matches_all_original_baselines(self):
        """Match baseline metrics and action counts without mutating inputs or preview state."""
        tasks = generate_tasks(seed=42)
        original = copy.deepcopy(tasks)
        for policy in (*ACTIONS, 'Random', 'Greedy'):
            with self.subTest(policy=policy):
                expected_delay, expected_success, expected_counts = evaluate_policy(policy, tasks)
                env = LearningEnvironment(tasks)
                rng = Random(7)
                delays, successes = [], 0
                counts = dict.fromkeys(ACTIONS, 0)
                for task in tasks:
                    env.simulator.uplink_mbps = task['uplink_mbps']
                    before = (env.simulator.time_s, env.simulator.server_free_at.copy())
                    location = choose_action(policy, env.simulator, task, rng)
                    self.assertEqual(before, (env.simulator.time_s, env.simulator.server_free_at))
                    _, _, _, info = env.step(ACTIONS.index(location))
                    delays.append(info['delay_s'])
                    successes += info['deadline_met']
                    counts[location] += 1
                self.assertAlmostEqual(float(np.mean(delays)), expected_delay)
                self.assertAlmostEqual(100 * successes / len(tasks), expected_success)
                self.assertEqual(counts, expected_counts)
        self.assertEqual(tasks, original)


class AgentTests(unittest.TestCase):
    def make_agent(self):
        return DQNAgent(seed=7, settings={
            'batch_size': 2, 'warmup_steps': 2, 'train_every': 1,
            'replay_capacity': 10, 'target_update_steps': 1000, 'gamma': 0.9,
        })

    def configured_loss(self, done):
        """Make targets analytically known; check online updates and gradient-free targets.

        Predicted values start at zero: terminal targets are -2, while continuing
        targets are -2 + 0.9 * 10 = 7. Smooth L1 losses are therefore 1.5 and 6.5.
        """
        agent = self.make_agent()
        with torch.no_grad():
            for parameter in agent.network.parameters():
                parameter.zero_()
            for parameter in agent.target_network.parameters():
                parameter.zero_()
            agent.target_network.layers[-1].bias.fill_(10.0)
        state = np.zeros(8, dtype=np.float32)
        for _ in range(2):
            agent.remember(state, 0, -2.0, state, done)
        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertNotEqual(agent.network.layers[-1].bias[0].item(), 0.0)
        self.assertTrue(all(p.grad is None for p in agent.target_network.parameters()))
        return loss

    def test_terminal_target_omits_future_value(self):
        self.assertAlmostEqual(self.configured_loss(True), 1.5, places=5)

    def test_nonterminal_target_includes_discounted_value(self):
        self.assertAlmostEqual(self.configured_loss(False), 6.5, places=5)

    def test_replay_copies_observations(self):
        """Later caller mutations must not alter either state saved in replay."""
        agent = self.make_agent()
        state = np.ones(8, dtype=np.float32)
        agent.remember(state, 1, -0.2, state, False)
        state[:] = 99
        np.testing.assert_array_equal(agent.memory[0][0], np.ones(8))
        np.testing.assert_array_equal(agent.memory[0][3], np.ones(8))

    def test_checkpoint_round_trip(self):
        """Serialization preserves exact action values, greedy choice, and metadata."""
        agent = self.make_agent()
        state = np.arange(8, dtype=np.float32) / 8
        tensor = torch.as_tensor(state).unsqueeze(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            agent.save(path, metadata={'purpose': 'test'})
            loaded = DQNAgent.load(path)
            with torch.no_grad():
                torch.testing.assert_close(agent.network(tensor), loaded.network(tensor), rtol=0, atol=0)
            self.assertEqual(agent.select_action(state), loaded.select_action(state))
            self.assertEqual(loaded.metadata['purpose'], 'test')


if __name__ == '__main__':
    unittest.main()
