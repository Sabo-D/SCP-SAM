import torch
import torch.nn as nn
import torch.nn.functional as F


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
        dilation=1,
        groups=1,
        act=True,
        num_groups=8,
    ):
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(dilation * (k // 2) for k in kernel_size)
            else:
                padding = dilation * (kernel_size // 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )

        self.norm = make_group_norm(out_channels, num_groups=num_groups)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class StructureConditionGenerator(nn.Module):
    """
    Generate structure-conditioned positive / negative responses
    and scale selection weights.

    Input:
        fd: [B, Cd, Hd, Wd]

    Outputs:
        pos_response:  [B, 1, H, W]
        neg_response:  [B, 1, H, W]
        scale_weights: [B, K, 1, 1]
    """

    def __init__(
        self,
        struct_channels,
        sem_channels,
        num_scales=3,
        reduction=4,
    ):
        super().__init__()

        hidden = max(sem_channels // reduction, 16)

        self.align = nn.Sequential(
            ConvGNAct(
                struct_channels,
                sem_channels,
                kernel_size=1,
                padding=0,
            ),
            ConvGNAct(
                sem_channels,
                sem_channels,
                kernel_size=3,
                groups=sem_channels,
            ),
            ConvGNAct(
                sem_channels,
                sem_channels,
                kernel_size=1,
                padding=0,
            ),
        )

        self.response_head = nn.Sequential(
            ConvGNAct(
                sem_channels,
                hidden,
                kernel_size=1,
                padding=0,
            ),
            ConvGNAct(
                hidden,
                hidden,
                kernel_size=3,
                groups=hidden,
            ),
            nn.Conv2d(
                hidden,
                2,
                kernel_size=1,
                bias=True,
            ),
        )

        self.scale_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                sem_channels,
                hidden,
                kernel_size=1,
                bias=True,
            ),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                num_scales,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, fd, target_size):
        if fd.shape[-2:] != target_size:
            fd = F.interpolate(
                fd,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        fd = self.align(fd)

        response_logits = self.response_head(fd)
        response_prob = torch.softmax(response_logits, dim=1)

        pos_response = response_prob[:, 0:1]
        neg_response = response_prob[:, 1:2]

        scale_logits = self.scale_head(fd)
        scale_weights = torch.softmax(scale_logits, dim=1)

        return pos_response, neg_response, scale_weights

class StructureAwareContextMixer(nn.Module):
    """
    Structure-aware multi-scale semantic context mixer.

    Input:
        fs:            [B, C, H, W]
        scale_weights: [B, K, 1, 1]

    Output:
        f_ctx: [B, C, H, W]
    """

    def __init__(
        self,
        channels,
        dilations=(1, 2, 3),
    ):
        super().__init__()

        self.num_scales = len(dilations)
        self.branches = nn.ModuleList()

        for dilation in dilations:
            self.branches.append(
                nn.Sequential(
                    ConvGNAct(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=channels,
                    ),
                    ConvGNAct(
                        channels,
                        channels,
                        kernel_size=1,
                        padding=0,
                    ),
                )
            )

    def forward(self, fs, scale_weights):
        f_ctx = None

        for idx, branch in enumerate(self.branches):
            feat = branch(fs)
            weight = scale_weights[:, idx:idx + 1]
            weighted_feat = weight * feat
            f_ctx = weighted_feat if f_ctx is None else f_ctx + weighted_feat

        return f_ctx

class PrototypeContrastRecalibrator(nn.Module):
    """
    Prototype contrast guided residual recalibration.

    Inputs:
        fs:           [B, C, H, W]
        f_ctx:        [B, C, H, W]
        pos_response: [B, 1, H, W]
        neg_response: [B, 1, H, W]

    Output:
        delta: [B, C, H, W]
    """

    def __init__(
        self,
        channels,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.fg_transform = nn.Sequential(
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

        self.bg_transform = nn.Sequential(
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

        self.out_proj = nn.Sequential(
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

    def weighted_avg_pool(self, feat, weight):
        numerator = (feat * weight).sum(
            dim=(2, 3),
            keepdim=True,
        )
        denominator = weight.sum(
            dim=(2, 3),
            keepdim=True,
        ) + self.eps

        return numerator / denominator

    def forward(
        self,
        fs,
        f_ctx,
        pos_response,
        neg_response,
    ):
        pos_proto = self.weighted_avg_pool(
            f_ctx,
            pos_response,
        )

        neg_proto = self.weighted_avg_pool(
            f_ctx,
            neg_response,
        )

        f_norm = F.normalize(
            f_ctx,
            dim=1,
        )
        pos_proto_norm = F.normalize(
            pos_proto,
            dim=1,
        )
        neg_proto_norm = F.normalize(
            neg_proto,
            dim=1,
        )

        sim_pos = (
            f_norm * pos_proto_norm
        ).sum(dim=1, keepdim=True)

        sim_neg = (
            f_norm * neg_proto_norm
        ).sum(dim=1, keepdim=True)

        contrast = torch.sigmoid(
            sim_pos - sim_neg
        )

        pos_gate = pos_response * contrast
        neg_gate = neg_response * (1.0 - contrast)

        fg_feature = self.fg_transform(f_ctx)
        bg_feature = self.bg_transform(fs)

        delta = (
            pos_gate * fg_feature
            - neg_gate * bg_feature
        )

        delta = self.out_proj(delta)

        return delta

class StructureConditionedPrototypeRecalibrator(nn.Module):
    """
    SCPR: Structure-conditioned Prototype Recalibrator.

    Position:
        Between SAM Image Encoder and SAM Mask Decoder.

    Inputs:
        fs:
            SAM image encoder semantic feature.
            Shape: [B, C, H, W]

        fd:
            Structure feature from SRA, usually d16.
            Shape: [B, Cd, Hd, Wd]

    Output:
        out:
            Recalibrated semantic feature.
            Shape: [B, C, H, W]

    Design:
        1. Generate positive / negative structure responses from fd.
        2. Generate structure-guided scale weights from fd.
        3. Mix multi-scale semantic contexts from fs.
        4. Build positive / negative prototypes from structure-aware context.
        5. Use prototype contrast to produce positive / negative gates.
        6. Inject residual recalibration into fs.
    """

    def __init__(
        self,
        sem_channels=256,
        struct_channels=48,
        dilations=(1, 2, 3),
        reduction=4,
        gamma_init=0.01,
    ):
        super().__init__()

        self.num_scales = len(dilations)

        self.condition_generator = StructureConditionGenerator(
            struct_channels=struct_channels,
            sem_channels=sem_channels,
            num_scales=self.num_scales,
            reduction=reduction,
        )

        self.context_mixer = StructureAwareContextMixer(
            channels=sem_channels,
            dilations=dilations,
        )

        self.prototype_recalibrator = PrototypeContrastRecalibrator(
            channels=sem_channels,
        )

        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )

    def forward(self, fs, fd):
        target_size = fs.shape[-2:]

        pos_response, neg_response, scale_weights = self.condition_generator(
            fd=fd,
            target_size=target_size,
        )

        f_ctx = self.context_mixer(
            fs=fs,
            scale_weights=scale_weights,
        )

        delta = self.prototype_recalibrator(
            fs=fs,
            f_ctx=f_ctx,
            pos_response=pos_response,
            neg_response=neg_response,
        )

        out = fs + self.gamma * delta

        return out, pos_response, neg_response
