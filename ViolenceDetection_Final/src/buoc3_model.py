from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.models.mobilenetv3 as mobilenetv3


class TemporalShift(nn.Module):
    def __init__(self, net: nn.Module, n_segment: int = 8, n_div: int = 8) -> None:
        super().__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shift(x, self.n_segment, fold_div=self.fold_div)
        return self.net(x)

    @staticmethod
    def shift(x: torch.Tensor, n_segment: int, fold_div: int = 8) -> torch.Tensor:
        nt, channels, height, width = x.size()
        if nt % n_segment != 0:
            raise ValueError(f"Input batch*time ({nt}) is not divisible by n_segment ({n_segment})")

        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, channels, height, width)

        fold = channels // fold_div
        if fold <= 0:
            return x.view(nt, channels, height, width)

        fold2 = min(fold * 2, channels)
        out = torch.zeros_like(x)

        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:fold2] = x[:, :-1, fold:fold2]
        out[:, :, fold2:] = x[:, :, fold2:]

        return out.view(nt, channels, height, width)


def make_temporal_shift(
    model: nn.Module,
    n_segment: int,
    n_div: int = 8,
    shift_stride: int = 2,
) -> int:
    if shift_stride < 1:
        raise ValueError("shift_stride must be >= 1")

    inserted = 0
    ir_blocks = [m for m in model.modules() if isinstance(m, mobilenetv3.InvertedResidual)]

    for block_index, block in enumerate(ir_blocks):
        if block_index % shift_stride != 0:
            continue

        if len(block.block) == 0:
            continue

        block.block[0] = TemporalShift(block.block[0], n_segment=n_segment, n_div=n_div)
        inserted += 1

    return inserted


class MobileNetV3TSMLite(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        num_segments: int = 8,
        pretrained: bool = True,
        shift_div: int = 8,
        shift_stride: int = 2,
        dropout: float = 0.20,
        variant: str = "small",
    ) -> None:
        super().__init__()

        self.num_segments = num_segments
        self.variant = variant

        if self.variant == "large":
            weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            self.base_model = tv_models.mobilenet_v3_large(weights=weights)
        else:
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.base_model = tv_models.mobilenet_v3_small(weights=weights)

        self.temporal_block_count = make_temporal_shift(
            self.base_model,
            n_segment=num_segments,
            n_div=shift_div,
            shift_stride=shift_stride,
        )

        if isinstance(self.base_model.classifier[-2], nn.Dropout):
            self.base_model.classifier[-2].p = dropout

        in_features = self.base_model.classifier[-1].in_features
        self.base_model.classifier[-1] = nn.Linear(in_features, num_classes)

    def freeze_backbone(self) -> None:
        for parameter in self.base_model.features.parameters():
            parameter.requires_grad = False

    def unfreeze_all(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad = True

    def count_trainable_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x: torch.Tensor, return_frame_logits: bool = False):
        if x.ndim != 5:
            raise ValueError(f"Expected input tensor [B, T, C, H, W], got shape {tuple(x.shape)}")

        batch, steps, channels, height, width = x.shape
        if steps != self.num_segments:
            raise ValueError(
                f"Input steps ({steps}) must match model num_segments ({self.num_segments})."
            )

        x = x.reshape(batch * steps, channels, height, width)
        frame_logits = self.base_model(x)
        frame_logits = frame_logits.view(batch, steps, -1)

        video_logits = frame_logits.mean(dim=1)
        if return_frame_logits:
            return video_logits, frame_logits
        return video_logits


class MobileNetV3_TSM(MobileNetV3TSMLite):
    def __init__(self, num_classes: int = 2, num_segments: int = 8, pretrained: bool = True, variant: str = "small"):
        super().__init__(
            num_classes=num_classes,
            num_segments=num_segments,
            pretrained=pretrained,
            shift_div=8,
            shift_stride=2,
            dropout=0.20,
            variant=variant
        )


def create_tsm_model(
    num_classes: int = 2,
    num_segments: int = 8,
    pretrained: bool = True,
    shift_div: int = 8,
    shift_stride: int = 2,
    dropout: float = 0.20,
) -> MobileNetV3TSMLite:
    return MobileNetV3TSMLite(
        num_classes=num_classes,
        num_segments=num_segments,
        pretrained=pretrained,
        shift_div=shift_div,
        shift_stride=shift_stride,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = create_tsm_model(num_classes=2, num_segments=8, pretrained=False)
    dummy = torch.randn(2, 8, 3, 224, 224)
    logits, frame_logits = model(dummy, return_frame_logits=True)

    print("video logits:", tuple(logits.shape))
    print("frame logits:", tuple(frame_logits.shape))
    print("temporal blocks:", model.temporal_block_count)
    print("trainable params:", model.count_trainable_parameters())
