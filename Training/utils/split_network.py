import torch
from torch import nn


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.visual = model.visual

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image):
        return self.visual(image.type(self.dtype))


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.out_proj.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


class ImageUnicl(nn.Module):
    def __init__(self, model):
        super(ImageUnicl, self).__init__()
        self.image_encoder = model.image_encoder
        self.image_projection = model.image_projection

    # @property
    # def dtype(self):
    #     return self.visual.conv1.weight.dtype

    def forward(self, image):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        return x


class TextUnicl(nn.Module):
    def __init__(self, model):
        super(TextUnicl, self).__init__()
        self.text_encoder = model.text_encoder
        self.text_projection = model.text_projection

    # @property
    # def dtype(self):
    #     return self.transformer.resblocks[0].attn.out_proj.weight.dtype

    def forward(self, text):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']
        x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        x = x @ self.text_projection
        return x



def set_logit_scale():
    # def create_logits(x1, x2, logit_scale=model.logit_scale.float().exp()):
    def create_logits(x1, x2, logit_scale=100):
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_x1 = logit_scale*x1 @ x2.t()
        logits_per_x2 = logit_scale*x2 @ x1.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_x1, logits_per_x2
    return create_logits
