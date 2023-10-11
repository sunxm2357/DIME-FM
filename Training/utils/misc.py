import os
import shutil
import torch


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_log_folder(args, model_name):
    teacher_name = 'ViT-L-14'
    log_folder = '%s-%s' % (model_name, teacher_name)
    log_folder += '-b%d' % args.batch_size
    if args.prefix:
        log_folder += '-%s' % args.prefix
    log_folder += '-e%d' % args.epochs
    log_folder = os.path.join(args.output_dir, log_folder)
    return log_folder


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))