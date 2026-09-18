import unittest

import torch

from speaker_id.models import create_classifier


class ModelTests(unittest.TestCase):
    def test_resnet_output_shape(self) -> None:
        model = create_classifier("resnet18", 24)
        output = model(torch.randn(2, 1, 64, 64))
        self.assertEqual(output.shape, (2, 24))
