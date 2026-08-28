import unittest

from terraclass.data import stratified_split_indices


class StratifiedSplitTests(unittest.TestCase):
    def setUp(self):
        self.targets = [0] * 10 + [1] * 20 + [2] * 30

    def test_splits_are_disjoint_and_complete(self):
        train, validation, test = stratified_split_indices(
            self.targets,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=7,
        )

        self.assertEqual(len(train), 36)
        self.assertEqual(len(validation), 12)
        self.assertEqual(len(test), 12)
        self.assertEqual(set(train) | set(validation) | set(test), set(range(60)))
        self.assertFalse(set(train) & set(validation))
        self.assertFalse(set(train) & set(test))
        self.assertFalse(set(validation) & set(test))

    def test_split_is_repeatable_for_the_same_seed(self):
        first = stratified_split_indices(self.targets, seed=42)
        second = stratified_split_indices(self.targets, seed=42)
        different = stratified_split_indices(self.targets, seed=43)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_invalid_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            stratified_split_indices(self.targets, val_ratio=0.6, test_ratio=0.4)


if __name__ == "__main__":
    unittest.main()
