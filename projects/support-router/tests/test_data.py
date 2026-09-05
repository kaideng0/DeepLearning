import unittest
from collections import Counter

from support_router.data import TextIntentDataset, stratified_train_validation_indices


class DataTests(unittest.TestCase):
    def test_dataset_validates_lengths(self) -> None:
        with self.assertRaises(ValueError):
            TextIntentDataset(["one"], [0, 1])

    def test_stratified_split_is_complete_disjoint_and_reproducible(self) -> None:
        targets = [class_id for class_id in range(3) for _ in range(10)]
        first = stratified_train_validation_indices(targets, validation_ratio=0.2, seed=7)
        second = stratified_train_validation_indices(targets, validation_ratio=0.2, seed=7)
        train_indices, validation_indices = first

        self.assertEqual(first, second)
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(set(train_indices) | set(validation_indices), set(range(30)))
        validation_counts = Counter(targets[index] for index in validation_indices)
        self.assertEqual(validation_counts, Counter({0: 2, 1: 2, 2: 2}))


if __name__ == "__main__":
    unittest.main()
