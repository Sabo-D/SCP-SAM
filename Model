import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from src.new.Encoder import PVSAMImageEncoder, StructureInjectionAdapter
from src.new.PSE import PhotovoltaicStructureEncoder
from src.new.SRA import StructureRefinementAdapter
from src.new.SCPR import StructureConditionedPrototypeRecalibrator
from src.new.BSH import BoundaryStructureHead

class SCPSAM(nn.Module):
    def __init__(
        self,
        sam_checkpoint,
        model_type="vit_b",
        image_size=1024,
        base_channels=48,
        inject_layers=None,
        prompt_alpha_init=0.01,
        freeze_mask_decoder=True,
        freeze_prompt_encoder=True,
        use_scpr=True,
    ):
        super().__init__()

        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.image_size = image_size
        self.use_scpr = use_scpr

        embed_dim = self.sam.image_encoder.patch_embed.proj.out_channels
        patch_size = self.sam.image_encoder.patch_embed.proj.kernel_size[0]

        # 1. PSE: image_rgb -> b4, b16
        self.structure_encoder = PhotovoltaicStructureEncoder(
            in_channels=3,
            base_channels=base_channels,
            kernel_sizes=(5, 9, 15),
            use_abs_prior=False,
        )

        # 2. SRA: b4 -> d4, d16
        self.sra = StructureRefinementAdapter(
            channels=base_channels,
            gamma_init=0.01,
        )

        # 3. d16 -> ViT prompt
        self.injection_adapter = StructureInjectionAdapter(
            structure_channels=base_channels,
            vit_channels=embed_dim,
        )

        # 4. SAM image encoder with prompt injection
        self.image_encoder = PVSAMImageEncoder(
            sam_image_encoder=self.sam.image_encoder,
            inject_layers=inject_layers,
            prompt_alpha_init=prompt_alpha_init,
        )

        # 5. SCPR: image_embeddings + d16 -> recalibrated image_embeddings
        if self.use_scpr:
            self.scpr = StructureConditionedPrototypeRecalibrator(
                sem_channels=256,
                struct_channels=base_channels,
                dilations=(1, 2, 3),
                reduction=4,
                gamma_init=0.01,
            )

        # 6. BSH:d4-->region_logits, boundary_logits
        self.boundary_head = BoundaryStructureHead(
            in_channels=base_channels,
            hidden_channels=32,
        )

        if freeze_mask_decoder:
            for p in self.sam.mask_decoder.parameters():
                p.requires_grad = False

        if freeze_prompt_encoder:
            for p in self.sam.prompt_encoder.parameters():
                p.requires_grad = False

    def forward(self, image_sam, image_rgb):
        # 1. PSE
        # b4:  [B, C, H/4,  W/4]
        # b16: [B, C, H/16, W/16]
        b4, b16, detail, var, aniso = self.structure_encoder(image_rgb)

        # 2. SRA
        # d4:  [B, C, H/4,  W/4]
        # d16: [B, C, H/16, W/16]
        d4, d16 = self.sra(b4)

        region_logits, boundary_logits = self.boundary_head(d4)
        boundary_logits = F.interpolate(
            boundary_logits,
            size=image_sam.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # 3. 用 PSE 的 b16 生成 prompt，注入 SAM image encoder
        prompt = self.injection_adapter(
            structure_feature=b16,
            target_hw=None,
        )

        # 4. prompt 注入 SAM image encoder
        image_embeddings = self.image_encoder(
            image_sam=image_sam,
            prompt=prompt,
        )

        # 5. 用 SRA 的 d16 做 SCPR 语义重校准
        if self.use_scpr:
            image_embeddings, pos, neg = self.scpr(
                fs=image_embeddings,
                fd=d16,
            )

        batch_size = image_sam.shape[0]

        sparse_embeddings = torch.empty(
            batch_size,
            0,
            self.sam.prompt_encoder.embed_dim,
            device=image_sam.device,
            dtype=image_embeddings.dtype,
        )

        dense_pe = self.sam.prompt_encoder.get_dense_pe()

        if dense_pe.shape[-2:] != image_embeddings.shape[-2:]:
            dense_pe = F.interpolate(
                dense_pe,
                size=image_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        dense_prompt_embeddings = torch.zeros_like(image_embeddings)

        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=False,
        )

        mask_logits = F.interpolate(
            low_res_masks,
            size=image_sam.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return mask_logits, boundary_logits
