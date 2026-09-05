import unittest

from support_router.cli import build_parser


class CliTests(unittest.TestCase):
    def test_predict_arguments(self) -> None:
        args = build_parser().parse_args(
            ["predict", "my card is missing", "--experiment-dir", "outputs/scratch"]
        )
        self.assertEqual(args.command, "predict")
        self.assertEqual(args.text, ["my card is missing"])


if __name__ == "__main__":
    unittest.main()
