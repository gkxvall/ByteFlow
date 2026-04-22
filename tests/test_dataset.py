from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from byteflow.dataset import (
    ByteFlowImageFolder,
    build_val_transforms,
    scan_image_folder,
    train_val_split,
)


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for cls, color in [("zebra", (1, 2, 3)), ("apple", (4, 5, 6))]:
            d = self.root / cls
            d.mkdir()
            img = Image.new("RGB", (32, 24), color)
            img.save(d / f"{cls}_0.png")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sorted_class_names_deterministic_mapping(self) -> None:
        samples, class_to_idx = scan_image_folder(self.root)
        self.assertEqual(class_to_idx, {"apple": 0, "zebra": 1})
        labels = {lbl for _, lbl in samples}
        self.assertEqual(labels, {0, 1})

    def test_indexing_and_label(self) -> None:
        samples, class_to_idx = scan_image_folder(self.root)
        tfm = build_val_transforms(224)
        ds = ByteFlowImageFolder(samples, class_to_idx, transform=tfm)
        x, y = ds[0]
        self.assertEqual(x.shape, (3, 224, 224))
        self.assertIsInstance(y, int)
        self.assertIn(y, (0, 1))

    def test_train_val_split_reproducible(self) -> None:
        samples, _ = scan_image_folder(self.root)
        a1, b1 = train_val_split(samples, 0.5, seed=123)
        a2, b2 = train_val_split(samples, 0.5, seed=123)
        self.assertEqual([s[0] for s in a1], [s[0] for s in a2])


if __name__ == "__main__":
    unittest.main()
