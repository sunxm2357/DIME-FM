import logging

import torch
from torch import nn

from timm.models.layers import DropPath, trunc_normal_


logger = logging.getLogger(__name__)


class BiProjModel(nn.Module):
    def __init__(self, config, is_image=True, is_text=True):
        super().__init__()
        assert is_image or is_text
        self.is_image = is_image
        self.is_text = is_text
        dim_projection = config.UNICL_MODEL.DIM_PROJECTION
        if is_text:
            dim_out = config.LANG_ENCODER.WIDTH
            # self.lang_projection = nn.Parameter(torch.empty(dim_out, dim_projection))
            self.lang_projection = torch.nn.Linear(dim_out, dim_projection, bias=False)
            trunc_normal_(self.lang_projection.weight, std=.02)

        if is_image:
            dim_out = config.IMAGE_ENCODER.SPEC.EMBED_DIM
            # self.image_projection = nn.Parameter(
            #     torch.empty(dim_out, dim_projection)
            # )
            self.image_projection = torch.nn.Linear(dim_out, dim_projection, bias=False)

            trunc_normal_(self.image_projection.weight, std=.02)

    def from_pretrained(self, pretrained='', pretrained_layers=[], verbose=True):
        pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @property
    def dtype(self):
        for name, param in self.named_parameters():
            # param = getattr(self, name)
            return param.dtype

    def encode_image(self, image, norm=True):
        if self.is_image:
            x = self.image_projection(image)

            if norm:
                x = x / x.norm(dim=-1, keepdim=True)

            return x
        else:
            raise ValueError('[encode_image]: The model does not contain image projector')

    def encode_text(self, text, norm=True):
        if self.is_text:
            # x = text @ self.lang_projection
            x = self.lang_projection(text)

            if norm:
                x = x / x.norm(dim=-1, keepdim=True)

            return x
        else:
            raise ValueError('[encode_text]: The model does not contain text projector')

    def forward(self, image=None, text=None):
        if text is None:
            features_image = self.encode_image(image)
            return features_image
        elif image is None:
            features_text = self.encode_text(text)
            return features_text
        else:
            features_image = self.encode_image(image)
            features_text = self.encode_text(text)
            return features_image, features_text


def build_bi_proj_model(config, is_image=True, is_text=True,  **kwargs):
    model = BiProjModel(config, is_image=is_image, is_text=is_text)

    return model


class combine_clip(nn.Module):
    def __init__(self, image_encoder, text_encoder, proj_model, pre_proj_text):
        super(combine_clip, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.proj_model = proj_model
        self.pre_proj_text = pre_proj_text

    @property
    def dtype(self):
        for name, param in self.named_parameters():
            return param.dtype

    def encode_image(self, image):
        image_feature = self.image_encoder(image=image)
        image_feature = self.proj_model(image=image_feature)
        image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)

        return image_feature

    def encode_text(self, text):
        text_feature = self.text_encoder(text=text)
        if self.pre_proj_text:
            text_feature = self.pre_proj_text(text=text_feature)
        text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
        return text_feature

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)
        return features_image, features_text, 100.




