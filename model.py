"""
model.py  --  Compact CNN baseline (Stage 3b)

Four convolution blocks (Conv -> BatchNorm -> ReLU -> MaxPool) with 32, 64, 128 and 256
filters, then global average pooling, dropout and one linear layer with 7 outputs.
The same architecture is used for CPU and GPU runs.

Run directly (python model.py) to print the model summary and check the output shape.
"""

import torch
import torch.nn as nn

import config


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SkinCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32),      # 128 -> 64
            conv_block(32, 64),     # 64  -> 32
            conv_block(64, 128),    # 32  -> 16
            conv_block(128, 256),   # 16  -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model():
    return SkinCNN()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = build_model()
    print(m)
    print(f"\nTrainable parameters: {count_parameters(m):,}")
    out = m(torch.randn(4, 3, config.IMAGE_SIZE, config.IMAGE_SIZE))
    print(f"Output shape for a batch of 4: {tuple(out.shape)} (expected (4, {config.NUM_CLASSES}))")
    print("MODEL CHECK PASSED")
