import torch
import torch.nn as nn


def make_group_norm(num_channels, num_groups=8):
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


class StructureRefinementAdapter(nn.Module):
    """
    SRA: Structure Refinement Adapter.

    Position:
        After Photovoltaic Structure Encoder.

    Input:
        b4: [B, C, H/4, W/4]

    Outputs:
        d4:  [B, C, H/4,  W/4]
        d16: [B, C, H/16, W/16]

    Usage:
        d4  -> BoundaryStructureHead
        d16 -> Structure-conditioned Prototype Recalibrator

    Design:
        Lightweight residual structure refinement and scale adaptation.
    """

    def __init__(
        self,
        channels=48,
        gamma_init=0.01,
    ):
        super().__init__()

        self.refine = nn.Sequential(
            ConvGNAct(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
                act=False,
            ),
        )

        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )

        self.down_to_16 = nn.Sequential(
            ConvGNAct(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=channels,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, b4):
        refined = self.refine(b4)

        d4 = b4 + self.gamma * refined
        d16 = self.down_to_16(d4)

        return d4, d16
