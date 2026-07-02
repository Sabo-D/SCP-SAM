import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from src.new.PSE import PhotovoltaicStructureEncoder, make_group_norm

class StructureInjectionAdapter(nn.Module):
    """
    Project structure feature b16 / d16 to SAM ViT token dimension.

    Input:
        structure_feature: [B, Cs, H, W]

    Output:
        prompt: [B, H, W, Cv]
    """

    def __init__(
        self,
        structure_channels=48,
        vit_channels=768,
    ):
        super().__init__()

        self.proj = nn.Conv2d(
            structure_channels,
            vit_channels,
            kernel_size=1,
            bias=False,
        )

        self.norm = nn.LayerNorm(vit_channels)

    def forward(self, structure_feature, target_hw=None):
        if target_hw is not None and structure_feature.shape[-2:] != target_hw:
            structure_feature = F.interpolate(
                structure_feature,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        prompt = self.proj(structure_feature)
        prompt = prompt.permute(0, 2, 3, 1).contiguous()
        prompt = self.norm(prompt)

        return prompt

class PVSAMImageEncoder(nn.Module):
    """
    Prompted SAM image encoder.

    Inputs:
        image_sam:
            SAM-normalized image, [B, 3, H, W]

        prompt:
            Structure prompt, [B, H_patch, W_patch, Cv]
            Usually generated from b16 / d16 by StructureInjectionAdapter.

    Output:
        image_embeddings:
            [B, 256, H_patch, W_patch]
    """

    def __init__(
        self,
        sam_image_encoder,
        inject_layers=None,
        prompt_alpha_init=0.01,
    ):
        super().__init__()

        self.sam_image_encoder = sam_image_encoder

        num_blocks = len(self.sam_image_encoder.blocks)

        if inject_layers is None:
            inject_layers = list(range(num_blocks))

        self.inject_layers = set(inject_layers)

        self.prompt_alpha = nn.Parameter(
            torch.tensor(float(prompt_alpha_init))
        )

    def forward(self, image_sam, prompt):
        x = self.sam_image_encoder.patch_embed(image_sam)

        # 保证 prompt 尺寸和 x 一致
        if prompt.shape[1:3] != x.shape[1:3]:
            prompt = prompt.permute(0, 3, 1, 2)
            prompt = F.interpolate(
                prompt,
                size=x.shape[1:3],
                mode="bilinear",
                align_corners=False,
            )
            prompt = prompt.permute(0, 2, 3, 1).contiguous()

        if self.sam_image_encoder.pos_embed is not None:
            pos_embed = self.sam_image_encoder.pos_embed

            if pos_embed.shape[1:3] != x.shape[1:3]:
                pos_embed = pos_embed.permute(0, 3, 1, 2)
                pos_embed = F.interpolate(
                    pos_embed,
                    size=x.shape[1:3],
                    mode="bilinear",
                    align_corners=False,
                )
                pos_embed = pos_embed.permute(0, 2, 3, 1)

            x = x + pos_embed

        for i, blk in enumerate(self.sam_image_encoder.blocks):
            if i in self.inject_layers:
                x = x + self.prompt_alpha * prompt

            x = blk(x)

        image_embeddings = self.sam_image_encoder.neck(
            x.permute(0, 3, 1, 2)
        )

        return image_embeddings
