import unittest

import torch

from terraclass.gradcam import create_gradcam
from terraclass.models import build_model, gradcam_target_layer


class ModelTests(unittest.TestCase):
    def test_cnn_output_shape(self):
        model = build_model("cnn", num_classes=10, pretrained=False)
        output = model(torch.randn(4, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (4, 10))

    def test_resnet_output_shape_without_downloading_weights(self):
        model = build_model("resnet18", num_classes=10, pretrained=False)
        model.eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (1, 10))

    def test_gradcam_matches_input_resolution(self):
        model = build_model("cnn", num_classes=3, pretrained=False)
        inputs = torch.randn(1, 3, 64, 64)
        heatmap, logits, class_index = create_gradcam(
            model,
            gradcam_target_layer(model, "cnn"),
            inputs,
        )

        self.assertEqual(tuple(heatmap.shape), (64, 64))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertIn(class_index, range(3))
        self.assertGreaterEqual(float(heatmap.min()), 0.0)
        self.assertLessEqual(float(heatmap.max()), 1.0)

    def test_gradcam_supports_resnet(self):
        model = build_model("resnet18", num_classes=3, pretrained=False)
        inputs = torch.randn(1, 3, 64, 64)
        heatmap, logits, class_index = create_gradcam(
            model,
            gradcam_target_layer(model, "resnet18"),
            inputs,
        )

        self.assertEqual(tuple(heatmap.shape), (64, 64))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertIn(class_index, range(3))


if __name__ == "__main__":
    unittest.main()
