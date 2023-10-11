from torch import optim as optim
from timm.scheduler.cosine_lr import CosineLRScheduler

def build_optimizer_and_scheduler(args, model, epochs,  n_iter_per_epoch):
    linear_scaled_lr = args.base_lr
    linear_scaled_warmup_lr = args.warmup_lr
    linear_scaled_min_lr = args.min_lr
    # gradient accumulation also need to scale the learning rate
    skip = []
    skip_keywords = ['relative_position_bias_table', 'absolute_pos_embed', 'positional_embedding', 'token_embedding']
    parameters = set_weight_decay(model, skip, skip_keywords)

    optimizer = optim.AdamW(parameters, eps=1e-8, betas=(0.9, 0.999),
                                lr=linear_scaled_lr, weight_decay=0.05)
    num_steps = int(epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
    if args.local_rank in [0, -1]:
        print('Initial lr: %.5f, warm-up lr: %.5f, min lr: %.5f' % (linear_scaled_lr, linear_scaled_warmup_lr,
                                                                    linear_scaled_min_lr), flush=True)
        print('# steps: %d, # warm-up steps: %d' % (num_steps, warmup_steps), flush=True)
    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=num_steps,
        lr_min=linear_scaled_min_lr,
        warmup_lr_init=linear_scaled_warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    return optimizer, lr_scheduler


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    # print('# params with decay = ', len(has_decay), flush=True)
    # print('# params without decay = ', len(no_decay), flush=True)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin