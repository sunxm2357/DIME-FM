import sys
import os
import shutil
import time
import torch.cuda

from utils.opts import arg_parser
from utils.split_network import ImageCLIP, set_logit_scale, ImageUnicl
from clip.model import CLIP
from unicl_model import UniCLModel
import yaml
from utils.optim import build_optimizer_and_scheduler
from utils.misc import get_log_folder, makedir, save_checkpoint
from torch.utils.tensorboard import SummaryWriter
from datasets.build import build_dataloader

def optimizer_to(optim, device):
    # move optimizer to device
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def _get_tsv_list(tsv_file_list):

    tsv_list = []
    for l in tsv_file_list:
        if l.endswith('.list'):
            with open(l, 'r') as f:
                tsv_list.extend([i.strip() for i in f])
        else:
            tsv_list.append(l)

    print(f'tsv list: {tsv_list}')

    return tsv_list


def main():
    parser = arg_parser()
    args = parser.parse_args()
    if args.amp:
        from utils.utils_amp import train
        scaler = torch.cuda.amp.GradScaler()
    else:
        from utils.utils import train
        scaler = None

    # TODO: rewrite datasets based on tsv files, 4 different dataloaders for image, text, image_feat, text_feat
    print('... starting to load large-scale image dataset', flush=True)
    tsv_files = _get_tsv_list(args.tsv_file_list)
    image_dataloader, text_dataloader = build_dataloader(tsv_list=tsv_files,
                                                         dataroot=args.dataroot,
                                                         batch_size_per_gpu=args.batch_size,
                                                         num_workers=args.num_workers,
                                                         pin_memory=True)

    dataloaders = {
        "img_train_dataloader": image_dataloader,
        "text_train_dataloader": text_dataloader,
    }

    args.n_step_per_epoch = len(image_dataloader)

    create_logits = set_logit_scale()
    # build student network
    with open(args.student_config_file, 'r') as file:
        student_config = yaml.safe_load(file)
    print('Student model config file:', flush=True)
    print(student_config, flush=True)
    if student_config['MODEL']['TYPE'] == 'CLIP':
        student_model = CLIP(embed_dim=student_config['MODEL']['EMBED_DIM'],
                             image_resolution=student_config['MODEL']['IMAGE_RESOLUTION'],
                             vision_layers=student_config['MODEL']['VISION_LAYERS'],
                             vision_width=student_config['MODEL']['VISION_WIDTH'],
                             vision_patch_size=student_config['MODEL']['VISION_PATCH_SIZE'],
                             context_length=student_config['MODEL']['CONTEXT_LENGTH'],
                             vocab_size=student_config['MODEL']['VOCAB_SIZE'],
                             transformer_width=student_config['MODEL']['TRANSFORMER_WIDTH'],
                             transformer_heads=student_config['MODEL']['TRANSFORMER_HEADS'],
                             transformer_layers=student_config['MODEL']['TRANSFORMER_LAYERS']
                         )
        student_model = student_model.float()
        student_img_model = ImageCLIP(student_model)

    elif student_config['MODEL']['TYPE'] == 'Unicl':
        student_model = UniCLModel(
            DIM_PROJECTION=student_config['MODEL']['DIM_PROJECTION'],
            EMBED_DIM=student_config['MODEL']['IMAGE_ENCODER']['SWIN']['EMBED_DIM'],
            DEPTHS=student_config['MODEL']['IMAGE_ENCODER']['SWIN']['DEPTHS'],
            NUM_HEADS=student_config['MODEL']['IMAGE_ENCODER']['SWIN']['NUM_HEADS'],
            WINDOW_SIZE=student_config['MODEL']['IMAGE_ENCODER']['SWIN']['WINDOW_SIZE'],
            DROP_PATH_RATE=student_config['MODEL']['IMAGE_ENCODER']['DROP_PATH_RATE'],
            VOCAB_SIZE=student_config['MODEL']['TEXT_ENCODER']['VOCAB_SIZE'],
            TEXT_WIDTH=student_config['MODEL']['TEXT_ENCODER']['WIDTH'],
            TEXT_LAYERS=student_config['MODEL']['TEXT_ENCODER']['LAYERS'],
            TEXT_HEADS=student_config['MODEL']['TEXT_ENCODER']['HEADS'])
        student_model = student_model.float()
        student_img_model = ImageUnicl(student_model)
    else:
        raise ValueError('Student Type %s is not supported' % student_config['MODEL']['TYPE'])

    # Logging
    log_folder = get_log_folder(args, student_config['MODEL']['NAME'])
    makedir(log_folder)
    writer = SummaryWriter(log_dir=log_folder)

    # Define optimizer
    print('Construct Image Encoder Optimizer and Scheduler: ', flush=True)
    img_optimizer, img_scheduler = build_optimizer_and_scheduler(args, student_img_model, args.epochs, len(image_dataloader))

    optimizers = {
        "img_optimizer": img_optimizer,
    }

    schedulers = {
        "img_scheduler": img_scheduler,
    }

    best_loss, is_best = 1e7, False

    if os.path.exists(os.path.join(log_folder, 'log.log')):
        shutil.copyfile(os.path.join(log_folder, 'log.log'), os.path.join(
            log_folder, 'log.log.{}'.format(int(time.time()))))
    logfile = open(os.path.join(log_folder, 'log.log'), 'w')
    command = " ".join(sys.argv)
    # print to screen
    print(command, flush=True)
    print(args, flush=True)
    print(student_img_model, flush=True)
    # print to log file
    print(command, file=logfile, flush=True)
    print(args, file=logfile, flush=True)
    print(student_img_model, file=logfile, flush=True)

    # move the network to devices
    for k, v in optimizers.items():
        if v is not None:
            optimizer_to(v, 'cuda:0')

    models = {"create_logits": create_logits,
              "student_img_model": student_img_model,
              }

    for k, v in models.items():
        if isinstance(v, torch.nn.Module):
            if torch.cuda.is_available():
                v = v.cuda()
            if torch.cuda.device_count() > 1:
                v = torch.nn.DataParallel(v)
            models[k] = v

    for epoch in range(args.epochs):
        if args.amp:
            batch_time, train_loss, train_img_loss, train_text_loss, train_align_loss, train_p_align_loss = train(
                models, dataloaders, optimizers, schedulers, scaler, args, epoch, writer)
        else:
            batch_time, train_loss, train_img_loss, train_text_loss, train_align_loss, train_p_align_loss = train(
                models, dataloaders, optimizers, schedulers, args, epoch, writer)

        print(
            'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tImage Loss: {:4.4f}\tText Loss: {:4.4f}\t'
            'Align Loss: {:4.4f}\tPaired Align Loss: {:4.4f}\tSpeed: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_loss, train_img_loss, train_text_loss, train_align_loss, train_p_align_loss,
                batch_time * 1000), flush=True)

        print(
            'Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tImage Loss: {:4.4f}\tText Loss: {:4.4f}\t'
            'Align Loss: {:4.4f}\tPaired Align Loss: {:4.4f}\tSpeed: {:.2f} ms/batch'.format(
                epoch + 1, args.epochs, train_loss, train_img_loss, train_text_loss, train_align_loss, train_p_align_loss,
                batch_time * 1000), file=logfile, flush=True)

        save_dict = {'epoch': epoch + 1,
                     'image_state_dict': models['student_img_model'].module.state_dict(),
                     'best_loss': best_loss,
                     }

        if args.amp:
            save_dict['scaler'] = scaler.state_dict()

        for k, v in optimizers.items():
            if v is not None:
                save_dict['%s_state_dict' % k] = v.state_dict()

        for k, v in schedulers.items():
            if v is not None:
                save_dict['%s_state_dict' % k] = v.state_dict()

        save_checkpoint(save_dict, is_best=False, filepath=log_folder)


if __name__ == '__main__':
    main()