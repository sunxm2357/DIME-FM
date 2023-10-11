import os

from transformers import CLIPTokenizer
from transformers import AutoTokenizer
from .transformer import lang_encoder


def build_lang_encoder(VOCAB_SIZE,  WIDTH, LAYERS, HEADS):
    return lang_encoder(VOCAB_SIZE,  WIDTH, LAYERS, HEADS)


def build_tokenizer(config_encoder):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if config_encoder['TOKENIZER'] == 'clip':
        pretrained_tokenizer = config_encoder.get(
            'PRETRAINED_TOKENIZER', 'openai/clip-vit-base-patch32'
        )
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    else:
        tokenizer = AutoTokenizer.from_pretrained(config_encoder['TOKENIZER'])

    return tokenizer
