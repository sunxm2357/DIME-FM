from .swin_transformer import SwinTransformer

def build_model(EMBED_DIM, DEPTHS, NUM_HEADS, WINDOW_SIZE, DROP_PATH_RATE):
    model = SwinTransformer(
        num_classes=0,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=EMBED_DIM,
        depths=DEPTHS,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=DROP_PATH_RATE,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
        )

    return model
