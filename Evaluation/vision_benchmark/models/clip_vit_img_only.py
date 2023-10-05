from collections import OrderedDict
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel

from timm.models.layers import DropPath, trunc_normal_


from vision_benchmark.utils.comm import comm
from .cls_swin import SwinTransformer
from vision_benchmark.datasets.languages.build import build_tokenizer
import clip

logger = logging.getLogger(__name__)

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import logging
import os
from collections import OrderedDict

import torch

from timm.models.layers import DropPath, trunc_normal_
from torch import nn

logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            attn_mask: torch.Tensor = None,
            drop_path: float = 0.0,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            attn_mask: torch.Tensor = None,
            drop_path: float = 0.0,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask, drop_path)
                for _ in range(layers)
            ]
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # if is_main_process():
            #     logger.info("=> init weight of Linear/Conv2d from trunc norm")
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # if is_main_process():
                #     logger.info("=> init bias of Linear/Conv2d to zeros")
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisualTransformer(nn.Module):
    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            drop_path: float = 0.0,
            num_classes: int = 1000,
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.sequence_length = (input_resolution // patch_size) ** 2 + 1

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.sequence_length, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_path=drop_path)

        self.ln_post = LayerNorm(width)

        self.head = nn.Linear(width, num_classes) if num_classes > 0 else nn.Identity()
        self.dim_out = width

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
            # if is_main_process():
            #     logger.info("=> init weight of Linear/Conv2d from trunc norm")
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # if is_main_process():
                #     logger.info("=> init bias of Linear/Conv2d to zeros")
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def from_pretrained(self, pretrained="", pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            logging.info(f"=> loading pretrained model {pretrained}")
            pretrained_dict = torch.load(
                pretrained, map_location="cpu"
            )
            if "model" in pretrained_dict:
                pretrained_dict = pretrained_dict["model"]

            self.from_state_dict(pretrained_dict, pretrained_layers, verbose)

    def from_state_dict(self, pretrained_dict, pretrained_layers=[], verbose=True):
        model_dict = self.state_dict()
        stripped_key = lambda x: x[14:] if x.startswith("visual.") else x

        pretrained_dict = {
            stripped_key(k): v.to(torch.float32)
            for k, v in pretrained_dict.items()
            if stripped_key(k) in model_dict.keys()
        }
        need_init_state_dict = {}
        for k, v in pretrained_dict.items():
            need_init = (
                    k.split(".")[0] in pretrained_layers or pretrained_layers[0] == "*"
            )

            if need_init:
                if verbose:
                    logger.info(f"=> init {k} from pretrained state dict")

                need_init_state_dict[k] = v
        self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"positional_embedding", "class_embedding"}

    def forward_features(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                self.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class CLIP_vit_img_student(nn.Module):
    def __init__(self, config: dict,):
        super().__init__()

        teacher_model, _ = clip.load('ViT-L/14', jit=False)
        teacher_model = teacher_model.float()
        self.transformer = teacher_model.transformer
        self.token_embedding = teacher_model.token_embedding
        self.positional_embedding = teacher_model.positional_embedding
        self.ln_final = teacher_model.ln_final
        self.text_projection = teacher_model.text_projection

        embed_dim = config['MODEL']['SPEC']['EMBED_DIM']
        spec_vision = config['MODEL']['SPEC']['VISION']
        self.visual = VisualTransformer(
            input_resolution=config['TRAIN']['IMAGE_SIZE'][0],
            num_classes=0,
            patch_size=spec_vision["PATCH_SIZE"],
            width=spec_vision["EMBED_DIM"],
            layers=spec_vision["DEPTHS"],
            heads=spec_vision["NUM_HEADS"],
            drop_path=spec_vision["DROP_PATH_RATE"],
        )
        self.vision_projection = nn.Parameter(
            torch.empty(self.visual.dim_out, embed_dim)
        )
        trunc_normal_(self.vision_projection, std=.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logger.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        for k in self.text.no_weight_decay():
            no_weight_decay.add('text.'+k)

        for k in self.visual.no_weight_decay():
            no_weight_decay.add('visual.'+k)

        return no_weight_decay

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    @property
    def dtype(self):
        for name, _ in self.named_parameters():
            param = getattr(self, name)
            return param.dtype

    def encode_image(self, image, norm=True):
        x = self.visual.forward_features(image)
        x = x @ self.vision_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x


    def encode_text(self, text, norm=True):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T



def get_zeroshot_model(config, **kwargs):
    model = CLIP_vit_img_student(config)

    if config['MODEL']['INIT_WEIGHTS']:
        model.init_weights(
            config['MODEL']['PRETRAINED'],
            ["visual", 'vision_projection'],
            config['VERBOSE']
        )

    return model
