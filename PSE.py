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

class DirectionalGapPrior(nn.Module):
    """
    Fixed directional gap prior.

    Input:
        image: [B, 3, H, W]
        target_size: (H_out, W_out)

    Output:
        prior_feature: [B, C, H_out, W_out]
    """

    def __init__(
        self,
        out_channels,
        use_abs_response=False,
    ):
        super().__init__()

        self.use_abs_response = use_abs_response

        gap_h = torch.tensor(
            [[0, 0, 0],
             [1, -2, 1],
             [0, 0, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        gap_v = torch.tensor(
            [[0, 1, 0],
             [0, -2, 0],
             [0, 1, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        gap_45 = torch.tensor(
            [[0, 0, 1],
             [0, -2, 0],
             [1, 0, 0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        gap_135 = torch.tensor(
            [[1, 0, 0],
             [0, -2, 0],
             [0, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        self.register_buffer(
            "gap_bank",
            torch.cat([gap_h, gap_v, gap_45, gap_135], dim=0),
        )
        # [out_channels, in_channels, kernel_h, kernel_w]

        self.proj = nn.Sequential(
            ConvGNAct(
                1,
                out_channels,
                kernel_size=1,
                padding=0,
            ),
            ConvGNAct(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    @staticmethod
    def rgb_to_gray(x):
        if x.size(1) == 1:
            return x

        if x.size(1) == 3:
            return (
                0.299 * x[:, 0:1]
                + 0.587 * x[:, 1:2]
                + 0.114 * x[:, 2:3]
            )

        return x.mean(dim=1, keepdim=True)

    def forward(self, image, target_size):
        image = image.float()
        gray = self.rgb_to_gray(image)

        if gray.max() > 2.0:
            gray = gray / 255.0

        response = F.conv2d(
            gray,
            self.gap_bank,
            padding=1,
        )

        if self.use_abs_response:
            response = torch.abs(response)
        else:
            response = F.relu(response)

        # [B, 1, H, W]
        prior_map = torch.max(response, dim=1, keepdim=True)[0]

        # 自身最大值归一化到[0-1]，[B 1 H W]
        prior_map = prior_map / (
            prior_map.amax(dim=(2, 3), keepdim=True) + 1e-6
        )

        prior_map = F.adaptive_max_pool2d(
            prior_map,
            output_size=target_size,
        )

        prior_feature = self.proj(prior_map)

        return prior_feature

class StructureDescriptor(nn.Module):
    """
    Generate structure descriptors from early feature.

    Descriptor channels:
        local detail
        local variance
        directional anisotropy

    Outputs:
        scale_weights: [B, K, H, W]
        reliability:   [B, 1, H, W]
    """

    def __init__(
        self,
        num_scales=3,
        hidden_channels=16,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.scale_head = nn.Sequential(
            nn.Conv2d(
                3,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                num_scales,
                kernel_size=1,
                bias=True,
            ),
        )

        self.reliability_head = nn.Sequential(
            nn.Conv2d(
                3,
                hidden_channels,
                kernel_size=1,
                bias=False,
            ),
            make_group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                hidden_channels,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(self, feat):
        avg = F.avg_pool2d(
            feat,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        avg_sq = F.avg_pool2d(
            feat * feat,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        local_detail = torch.abs(feat - avg)
        local_detail = local_detail.mean(dim=1, keepdim=True)

        local_var = torch.clamp(avg_sq - avg * avg, min=0.0)
        local_var = local_var.mean(dim=1, keepdim=True)

        dx = torch.abs(feat[:, :, :, 1:] - feat[:, :, :, :-1])
        dy = torch.abs(feat[:, :, 1:, :] - feat[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))

        dx = F.avg_pool2d(
            dx,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        dy = F.avg_pool2d(
            dy,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        dx = dx.mean(dim=1, keepdim=True)
        dy = dy.mean(dim=1, keepdim=True)

        anisotropy = torch.abs(dx - dy) / (dx + dy + self.eps)

        descriptor = torch.cat(
            [
                local_detail,
                local_var,
                anisotropy,
            ],
            dim=1,
        )

        scale_weights = torch.softmax(
            self.scale_head(descriptor),
            dim=1,
        )

        reliability = self.reliability_head(descriptor)

        return scale_weights, reliability, local_detail, local_var, anisotropy

class PeripheralCenterUnit(nn.Module):
    """
    Peripheral-center structure unit.

    P_k = DWConv_kx1(DWConv_1xk(x))
    C   = DWConv_3x3(x)

    R_k = GELU(BN(P_k - 0.5 * C))
    """

    def __init__(
        self,
        channels,
        kernel_size,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.peripheral_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )

        self.peripheral_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )

        self.center = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.norm = make_group_norm(channels)
        self.act = nn.GELU()

    def forward(self, x):
        peripheral = self.peripheral_v(
            self.peripheral_h(x)
        )

        center = self.center(x)

        out = peripheral - 0.5 * center
        out = self.norm(out)
        out = self.act(out)

        return out

class MultiScalePeripheralCenter(nn.Module):
    """
    Descriptor-guided multi-scale peripheral-center modeling.

    Input:
        feat:          [B, C, H, W]
        scale_weights: [B, K, H, W]

    Output:
        dynamic_feature: [B, C, H, W]
    """

    def __init__(
        self,
        channels,
        kernel_sizes=(5, 9, 15),
    ):
        super().__init__()

        self.units = nn.ModuleList([
            PeripheralCenterUnit(
                channels=channels,
                kernel_size=k,
            )
            for k in kernel_sizes
        ])

        self.proj = nn.Sequential(
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
                groups=channels,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, feat, scale_weights):
        out = None

        for idx, unit in enumerate(self.units):
            response = unit(feat)
            weight = scale_weights[:, idx:idx + 1]
            response = response * weight
            out = response if out is None else out + response

        out = self.proj(out)

        return out

class PriorDynamicFusion(nn.Module):
    """
    Fuse fixed directional prior and dynamic structure feature.

    F = prior + gate * reliability * dynamic
    """

    def __init__(
        self,
        channels,
        reduction=4,
    ):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.gate = nn.Sequential(
            ConvGNAct(
                channels * 2,
                hidden,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(
                hidden,
                1,
                kernel_size=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        self.proj = nn.Sequential(
            ConvGNAct(
                channels,
                channels,
                kernel_size=3,
                groups=channels,
            ),
            ConvGNAct(
                channels,
                channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(
        self,
        prior_feature,
        dynamic_feature,
        reliability,
    ):
        gate = self.gate(
            torch.cat(
                [
                    prior_feature,
                    dynamic_feature,
                ],
                dim=1,
            )
        )

        fused = prior_feature + gate * reliability * dynamic_feature
        fused = self.proj(fused)

        return fused

class PhotovoltaicStructureEncoder(nn.Module):
    """
    PSE: Photovoltaic Structure Encoder.

    Position:
        Structure path entrance.

        image:
            It is recommended to use unnormalized RGB image in [0, 1]
            or [0, 255].

    Outputs:
        b4:
            Early structure feature.
            Shape: [B, C, H/4, W/4]
            Usage: send to StructureRefinementAdapter.

        b16:
            Patch-level structure feature.
            Shape: [B, C, H/16, W/16]
            Usage: inject into SAM Image Encoder.

    Design:
        1. Stem extracts early structure feature at stride 4.
        2. Directional gap prior provides fixed structural bias.
        3. Structure descriptor predicts scale weights and reliability.
        4. Multi-scale peripheral-center branch models dynamic structure.
        5. Prior-dynamic fusion generates structure-enhanced b4.
        6. b4 is downsampled to b16 for ViT patch-level injection.
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=48,
        kernel_sizes=(5, 9, 15),
        use_abs_prior=False,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            ConvGNAct(
                in_channels,
                base_channels // 2,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                base_channels // 2,
                base_channels,
                kernel_size=3,
                stride=2,
            ),
            ConvGNAct(
                base_channels,
                base_channels,
                kernel_size=3,
                groups=base_channels,
            ),
            ConvGNAct(
                base_channels,
                base_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.prior_branch = DirectionalGapPrior(
            out_channels=base_channels,
            use_abs_response=use_abs_prior,
        )

        self.descriptor = StructureDescriptor(
            num_scales=len(kernel_sizes),
            hidden_channels=16,
        )

        self.dynamic_branch = MultiScalePeripheralCenter(
            channels=base_channels,
            kernel_sizes=kernel_sizes,
        )

        self.fusion = PriorDynamicFusion(
            channels=base_channels,
            reduction=4,
        )

        self.down_to_16 = nn.Sequential(
            ConvGNAct(
                base_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                groups=base_channels,
            ),
            ConvGNAct(
                base_channels,
                base_channels,
                kernel_size=1,
                padding=0,
            ),
            ConvGNAct(
                base_channels,
                base_channels,
                kernel_size=3,
                stride=2,
                groups=base_channels,
            ),
            ConvGNAct(
                base_channels,
                base_channels,
                kernel_size=1,
                padding=0,
            ),
        )

    def forward(self, image):
        stem_feature = self.stem(image)  # H/4
        target_size = stem_feature.shape[-2:]

        # B C H W
        prior_feature = self.prior_branch(
            image,
            target_size=target_size,
        )

        # B 3 H W and B 1 H W
        scale_weights, reliability, detail, var, aniso = self.descriptor(
            stem_feature
        )

        # B C H W
        dynamic_feature = self.dynamic_branch(
            feat=stem_feature,
            scale_weights=scale_weights,
        )

        b4 = self.fusion(
            prior_feature=prior_feature,
            dynamic_feature=dynamic_feature,
            reliability=reliability,
        )

        b16 = self.down_to_16(b4)

        return b4, b16, detail, var, aniso
