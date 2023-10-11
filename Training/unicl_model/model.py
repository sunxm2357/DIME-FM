import torch
from torch import nn

from timm.models.layers import trunc_normal_

from .image_encoder import build_image_encoder
from .text_encoder import build_text_encoder


class UniCLModel(nn.Module):
    def __init__(self, DIM_PROJECTION, EMBED_DIM, DEPTHS, NUM_HEADS, WINDOW_SIZE, DROP_PATH_RATE, VOCAB_SIZE, TEXT_WIDTH, TEXT_LAYERS, TEXT_HEADS):
        super().__init__()

        self.text_encoder = build_text_encoder(VOCAB_SIZE, TEXT_WIDTH, TEXT_LAYERS, TEXT_HEADS)

        dim_projection = DIM_PROJECTION
        if hasattr(self.text_encoder, 'dim_out'):
            dim_out = self.text_encoder.dim_out
        else:
            with torch.no_grad():
                dim_out = self.text_encoder(
                    torch.zeros(1,1).type(torch.LongTensor)
                )['last_hidden_state'].size(2)

        self.text_projection = nn.Parameter(torch.empty(dim_out, dim_projection))

        self.image_encoder = build_image_encoder(EMBED_DIM, DEPTHS, NUM_HEADS, WINDOW_SIZE, DROP_PATH_RATE)

        self.image_projection = nn.Parameter(
            torch.empty(self.image_encoder.dim_out, dim_projection)
        )

        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        trunc_normal_(self.image_projection, std=.02)

    @property
    def dtype(self):
        return self.logit_scale.dtype

    def encode_image(self, image, norm=False):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=False):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']

        x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]

        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        features_image = features_image / features_image.norm(dim=1, keepdim=True)
        features_text = features_text / features_text.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        T = self.logit_scale.exp()
        logits_per_image = T * features_image @ features_text.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text



