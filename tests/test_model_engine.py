from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from dataset import ByteFlowImageFolder, build_train_transforms, scan_image_folder
from engine import train_one_epoch, validate_one_epoch
from model import build_model


class TestModelEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        for cls in ("c0", "c1"):
            d = root / cls
            d.mkdir()
            Image.new("RGB", (40, 40), (10, 10, 10)).save(d / "a.png")
        samples, class_to_idx = scan_image_folder(root)
        tfm = build_train_transforms(64)
        self.ds = ByteFlowImageFolder(samples, class_to_idx, transform=tfm)
        self.num_classes = len(class_to_idx)
        self.device = torch.device("cpu")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_forward_one_batch(self) -> None:
        loader = DataLoader(self.ds, batch_size=2, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        model = build_model("resnet18", self.num_classes, pretrained=False)
        with torch.no_grad():
            logits = model(batch_x)
        self.assertEqual(logits.shape, (2, self.num_classes))

    def test_train_and_validate_smoke(self) -> None:
        loader = DataLoader(self.ds, batch_size=2, shuffle=True)
        model = build_model("resnet18", self.num_classes, pretrained=False).to(
            self.device
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        m_train = train_one_epoch(model, loader, crit, opt, self.device)
        m_val = validate_one_epoch(model, loader, crit, self.device)
        self.assertIn("loss", m_train)
        self.assertIn("accuracy", m_train)
        self.assertIn("loss", m_val)
        self.assertGreaterEqual(m_val["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
