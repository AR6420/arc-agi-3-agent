"""ResNet-tiny backbone + three heads for BC pretrain.

Backbone: 3-channel float input → conv3x3(32) → 4 residual blocks
[32, 64, 64, 128] with strides [1, 2, 2, 1] → (B, 128, 16, 16).

Heads:
    - action_type:  global-avg-pool → Linear(128, 8)
    - spatial:      (B, 128, 16, 16) → 2× upsample-conv → (B, 1, 64, 64) logits
    - frame_change: (gap_feat + action_onehot_8) → 64 → 1  (teacher-forced action in)

Param count: ~1.6M (well under 5M Phase 0c cap).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(c_out),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), inplace=True)


class Backbone(nn.Module):
    """Phase 2.5: scaled to ~3M params total (was ~560K).

    Channel ladder [64, 128, 192, 256] with double-block stages 2 and 3.
    Output: (B, 256, 16, 16).
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.b1 = ResBlock(64, 64, stride=1)         # 64x64
        self.b2a = ResBlock(64, 128, stride=2)       # 32x32
        self.b2b = ResBlock(128, 128, stride=1)
        self.b3a = ResBlock(128, 192, stride=2)      # 16x16
        self.b3b = ResBlock(192, 192, stride=1)
        self.b4 = ResBlock(192, 256, stride=1)       # 16x16
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2a(x); x = self.b2b(x)
        x = self.b3a(x); x = self.b3b(x)
        x = self.b4(x)
        return x  # (B, 256, 16, 16)


class ActionTypeHead(nn.Module):
    def __init__(self, feat_dim: int = 256, n_actions: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_actions)

    def forward(self, feat_pool: torch.Tensor) -> torch.Tensor:
        return self.fc(feat_pool)


class SpatialHead(nn.Module):
    """Upsamples (B, C_in, 16, 16) → (B, 1, 64, 64) logits."""

    def __init__(self, c_in: int = 256) -> None:
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(c_in, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.up1(feat)
        x = self.up2(x)
        return self.head(x).squeeze(1)  # (B, 64, 64)


class FrameChangeHead(nn.Module):
    """Teacher-forced action input during training (Stage 2 fine-tunes on synth)."""

    def __init__(self, feat_dim: int = 256, n_actions: int = 8) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feat_dim + n_actions, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, feat_pool: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat_pool, action_onehot], dim=-1)
        x = F.relu(self.fc1(x), inplace=True)
        return self.fc2(x).squeeze(-1)  # (B,)


class EliteModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.action_head = ActionTypeHead(self.backbone.out_channels)
        self.spatial_head = SpatialHead(self.backbone.out_channels)
        self.framechange_head = FrameChangeHead(self.backbone.out_channels)

    def forward(
        self,
        x: torch.Tensor,
        action_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        feat_pool = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 128)
        action_logits = self.action_head(feat_pool)
        spatial_logits = self.spatial_head(feat)
        out = {
            "feat": feat,
            "feat_pool": feat_pool,
            "action_logits": action_logits,
            "spatial_logits": spatial_logits,
        }
        if action_id is not None:
            onehot = F.one_hot(action_id, num_classes=8).float()
            out["framechange_logits"] = self.framechange_head(feat_pool, onehot)
        return out


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
