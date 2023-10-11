import sys
sys.path.insert(0, '../model_distill/')
import torch
import torch.nn as nn
import time
from .utils import step_update, AverageMeter, zero_grad, get_loss, reduce_mean


def scaler_step(objs, scaler):
    for k, v in objs.items():
        if v is not None:
            scaler.step(v)
    return objs


def train(models, dataloaders, optimizers, schedulers, scaler, args, epoch, writer, text_train_enumerator=None, paired_train_enumerator=None):
    losses = AverageMeter()
    vl_losses = AverageMeter()
    pvl_losses = AverageMeter()
    udist_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    if text_train_enumerator is None:
        text_train_enumerator = enumerate(dataloaders['text_train_dataloader'])
    models['student_img_model'].train()

    if args.local_rank == -1:
        kl_loss = nn.KLDivLoss(reduction="batchmean")
    else:
        kl_loss = nn.KLDivLoss(reduction="batchmean").cuda(args.local_rank)
    start_time = time.time()
    for idx, (images, image_feats) in enumerate(dataloaders["img_train_dataloader"]):
        zero_grad(optimizers)
        text_idx, (texts, text_feats) = next(text_train_enumerator)
        if text_idx == len(dataloaders['text_train_dataloader']) - 1:
            text_train_enumerator = enumerate(dataloaders['text_train_dataloader'])

        data_time.update(time.time() - start_time)
        start_time = time.time()

        with torch.cuda.amp.autocast():
            loss, vl_loss, pvl_loss, udist_loss = get_loss(models, images, image_feats, text_feats, args, kl_loss)

        scaler.scale(loss).backward()
        scaler_step(optimizers, scaler)
        scaler.update()

        step_update(schedulers, epoch * len(dataloaders["img_train_dataloader"]) + idx)

        if args.local_rank != -1:
            loss = reduce_mean(loss, args.nprocs)
            vl_loss = reduce_mean(vl_loss, args.nprocs) if vl_loss != 0 else 0
            pvl_loss = reduce_mean(pvl_loss, args.nprocs) if pvl_loss != 0 else 0
            udist_loss = reduce_mean(udist_loss, args.nprocs) if udist_loss != 0 else 0

        vl_losses.update(vl_loss.item() if vl_loss != 0 else 0)
        pvl_losses.update(pvl_loss.item() if pvl_loss != 0 else 0)
        udist_losses.update(udist_loss.item() if udist_loss != 0 else 0)
        losses.update(loss.item())

        batch_time.update(time.time() - start_time)
        start_time = time.time()
        if idx % args.print_freq == 0 and args.local_rank in [-1, 0]:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'VL Loss {vl_loss.val:.4f} ({vl_loss.avg:.4f})\t'
                  'P-VL Loss {pvl_loss.val:.4f} ({pvl_loss.avg:.4f})\t'
                  'Udist Loss {udist_loss.val:.4f} ({udist_loss.avg:.4f})'.format(
                epoch, idx, len(dataloaders['img_train_dataloader']), batch_time=batch_time, data_time=data_time,
                loss=losses, vl_loss=vl_losses, pvl_loss=pvl_losses, udist_loss=udist_losses), flush=True)
            writer.add_scalar('Train/total_loss', losses.avg, idx + args.n_step_per_epoch * epoch)
            writer.add_scalar('Train/vl_loss', vl_losses.avg, idx + args.n_step_per_epoch * epoch)
            writer.add_scalar('Train/pvl_loss', pvl_losses.avg, idx + args.n_step_per_epoch * epoch)
            writer.add_scalar('Train/udist_loss', udist_losses.avg, idx + args.n_step_per_epoch * epoch)

    return batch_time.avg, losses.avg, vl_losses.avg, pvl_losses.avg, udist_losses.avg

