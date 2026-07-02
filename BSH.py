import torch
import torch.nn as nn
import torch.nn.functional as F

def make_group_norm(num_channels, num_groups=8):
    """
    GroupNorm for small-batch training.

    Automatically adjusts num_groups so that num_channels % num_groups == 0.
    """
    num_groups = min(num_groups, num_channels)

    while num_channels % num_groups != 0:
        num_groups -= 1

    return nn.GroupNorm(
        num_groups=num_groups,
        num_channels=num_channels,
    )


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True,
        num_groups=8,
    ):
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.norm = make_group_norm(out_channels, num_groups=num_groups)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BoundaryStructureHead(nn.Module):
    """
    Boundary-Structure Head.

    Input:
        d4: [B, C, H/4, W/4]

    Outputs:
        region_logits:   [B, 1, H/4, W/4]
        boundary_logits: [B, 1, H/4, W/4]

    Current usage:
        boundary_logits -> boundary auxiliary loss
        region_logits   -> reserved, not used now
    """

    def __init__(
        self,
        in_channels=48,
        hidden_channels=32,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            ConvGNAct(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            ),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.region_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

        self.boundary_head = nn.Conv2d(
            hidden_channels,
            1,
            kernel_size=1,
            bias=True,
        )

    def forward(self, d4):
        feat = self.shared(d4)

        region_logits = self.region_head(feat)
        boundary_logits = self.boundary_head(feat)

        return region_logits, boundary_logits
