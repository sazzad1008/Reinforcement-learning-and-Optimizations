import unittest

from PPO import PPOConfig, train_ppo
from PPO.data import make_complex_dataset, train_test_split


class PPOProjectTests(unittest.TestCase):
    def test_dataset_split_sizes(self):
        dataset = make_complex_dataset(size=100, feature_dim=12, seed=1)
        train_set, test_set = train_test_split(dataset, 0.8)
        self.assertEqual(len(train_set.features), 80)
        self.assertEqual(len(test_set.features), 20)

    def test_training_achieves_baseline_accuracy(self):
        result = train_ppo(PPOConfig())
        self.assertGreaterEqual(result.train_accuracy, 0.25)
        self.assertGreaterEqual(result.test_accuracy, 0.25)


if __name__ == "__main__":
    unittest.main()
