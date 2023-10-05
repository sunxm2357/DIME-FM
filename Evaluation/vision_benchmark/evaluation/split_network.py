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

    def forward(self, image):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        return x


class TextUnicl(nn.Module):
    def __init__(self, model):
        super(TextUnicl, self).__init__()
        self.text_encoder = model.text_encoder
        self.text_projection = model.text_projection

    def forward(self, text):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']
        x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        x = x @ self.text_projection
        return x

class ImageUniclwithText(nn.Module):
    def __init__(self, img_model=None, text_model=None, model=None):
        super(ImageUniclwithText, self).__init__()
        assert (img_model is not None and text_model is not None) or model is not None
        if model is not None:
            self.image_encoder = model.image_encoder
            self.image_projection = model.image_projection
            self.text_encoder = model.text_encoder
            self.text_projection = model.text_projection
        else:
            self.image_encoder = img_model.image_encoder
            self.image_projection = img_model.image_projection
            self.text_encoder = text_model.text_encoder
            self.text_projection = text_model.text_projection

    def encode_text(self, text):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']
        x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        x = x @ self.text_projection
        return x

    def forward(self, image):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        return x


class ImageUniclwithTextClip(nn.Module):
    def __init__(self, img_model=None, text_model=None):
        super(ImageUniclwithTextClip, self).__init__()
        self.image_encoder = img_model.image_encoder
        self.image_projection = img_model.image_projection
        self.transformer = text_model.transformer
        self.token_embedding = text_model.token_embedding
        self.positional_embedding = text_model.positional_embedding
        self.ln_final = text_model.ln_final
        self.text_projection = text_model.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].attn.out_proj.weight.dtype

    def encode_text(self, text):
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

    def forward(self, image):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        return x


class TextClipwithImageUnicl(nn.Module):
    def __init__(self, img_model=None, text_model=None):
        super(TextClipwithImageUnicl, self).__init__()
        self.image_encoder = img_model.image_encoder
        self.image_projection = img_model.image_projection
        self.transformer = text_model.transformer
        self.token_embedding = text_model.token_embedding
        self.positional_embedding = text_model.positional_embedding
        self.ln_final = text_model.ln_final
        self.text_projection = text_model.text_projection

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

    def encode_text(self, text):
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

    def encode_image(self, image):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        return x


class TextUniclwithImage(nn.Module):
    def __init__(self, img_model=None, text_model=None, model=None):
        super(TextUniclwithImage, self).__init__()
        assert (img_model is not None and text_model is not None) or model is not None
        if model is not None:
            self.image_encoder = model.image_encoder
            self.image_projection = model.image_projection
            self.text_encoder = model.text_encoder
            self.text_projection = model.text_projection
        else:
            self.image_encoder = img_model.image_encoder
            self.image_projection = img_model.image_projection
            self.text_encoder = text_model.text_encoder
            self.text_projection = text_model.text_projection

    def encode_image(self, image):
        x = self.image_encoder.forward_features(image)
        x = x @ self.image_projection
        return x

    def forward(self, text):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']
        x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        x = x @ self.text_projection
        return x


class TextUniclwithImageClip(nn.Module):
    def __init__(self, img_model=None, text_model=None):
        super(TextUniclwithImageClip, self).__init__()
        self.visual = img_model.visual
        self.text_encoder = text_model.text_encoder
        self.text_projection = text_model.text_projection

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def forward(self, text):
        x = self.text_encoder(**text)
        x = x['last_hidden_state']
        x = x[torch.arange(x.size(0)), text['input_ids'].argmax(dim=-1)]
        x = x @ self.text_projection
        return x


class ImageCLIPwText(nn.Module):
    def __init__(self, img_model=None, text_model=None, model=None):
        super(ImageCLIPwText, self).__init__()
        assert (img_model is not None and text_model is not None) or model is not None
        if model is not None:
            self.context_length = model.context_length
            self.visual = model.visual
            self.transformer = model.transformer
            self.vocab_size = model.vocab_size
            self.token_embedding = model.token_embedding
            self.positional_embedding = model.positional_embedding
            self.ln_final = model.ln_final
            self.text_projection = model.text_projection
            self.logit_scale = model.logit_scale
        else:
            self.visual = img_model.visual
            self.transformer = text_model.transformer
            self.token_embedding = text_model.token_embedding
            self.positional_embedding = text_model.positional_embedding
            self.ln_final = text_model.ln_final
            self.text_projection = text_model.text_projection

    def encode_text(self, text):
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

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def forward(self, image):
        return self.visual(image.type(self.dtype))


class TextCLIPwImage(nn.Module):
    def __init__(self, img_model=None, text_model=None, model=None):
        super(TextCLIPwImage, self).__init__()
        assert (img_model is not None and text_model is not None) or model is not None
        if model is not None:
            self.context_length = model.context_length
            self.visual = model.visual
            self.transformer = model.transformer
            self.vocab_size = model.vocab_size
            self.token_embedding = model.token_embedding
            self.positional_embedding = model.positional_embedding
            self.ln_final = model.ln_final
            self.text_projection = model.text_projection
            self.logit_scale = model.logit_scale
        else:
            self.visual = img_model.visual
            self.transformer = text_model.transformer
            self.token_embedding = text_model.token_embedding
            self.positional_embedding = text_model.positional_embedding
            self.ln_final = text_model.ln_final
            self.text_projection = text_model.text_projection

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

    def encode_text(self, text):
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

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))