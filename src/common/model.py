"""Backbone + classifier head shared by all three OncoAI tasks."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")      # Apple silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class OncoNet(nn.Module):
    """ImageNet-pretrained backbone with a small custom head.

    ResNet-18 is the default: these datasets are in the hundreds-to-thousands
    range, where a heavier backbone mostly buys overfitting. The head keeps a
    512-d bottleneck so embeddings can be reused for downstream fusion, mirroring
    the MammoAI Stage 2/3 design.
    """

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.resnet18(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            in_features = net.classifier[1].in_features
            net.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone {backbone!r}")

        self.features = net
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, return_embedding: bool = False):
        feat = self.features(x)
        if return_embedding:
            # 512-d activation from inside the head, for fusion experiments
            emb = self.head[:3](feat)
            return self.head[3:](emb), emb
        return self.head(feat)

    def freeze_backbone(self, frozen: bool = True) -> None:
        for p in self.features.parameters():
            p.requires_grad = not frozen

    def target_layer(self) -> nn.Module:
        """Last conv block — the layer GradCAM hooks."""
        if self.backbone_name == "resnet18":
            return self.features.layer4[-1]
        return self.features.features[-1]


def save_checkpoint(path, model: OncoNet, class_names: list[str], meta: dict) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": model.num_classes,
        "backbone": model.backbone_name,
        "class_names": class_names,
        "meta": meta,
    }, path)


def load_checkpoint(path, device=None):
    device = device or get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = OncoNet(ckpt["num_classes"], backbone=ckpt.get("backbone", "resnet18"),
                    pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
