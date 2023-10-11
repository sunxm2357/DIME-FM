import torch.nn.functional as F
import torch.nn as nn
import time
import torch.distributed as dist


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def zero_grad(optimizers):
    for k, v in optimizers.items():
        if v is not None:
            v.zero_grad()
    return optimizers


def step(objs):
    for k, v in objs.items():
        if v is not None:
            v.step()
    return objs


def step_update(objs, step):
    for k, v in objs.items():
        if v is not None:
            v.step_update(step)
    return objs


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_kl_loss(kl_loss_fn, teacher_logits, student_logits, temp):
    target = F.softmax(teacher_logits / temp, dim=1)
    input = F.log_softmax(student_logits / temp, dim=1)
    loss = kl_loss_fn(input, target)
    return loss


def get_loss(models, images, image_feats, text_feats, args, kl_loss):
    # move inputs to the device
    images = images.cuda(non_blocking=True)
    image_feats = image_feats.cuda( non_blocking=True)
    text_feats = text_feats.cuda( non_blocking=True)

    student_img_feat_sm = models['student_img_model'](images)

    # gather feature from all processes
    student_img_feat = student_img_feat_sm
    device = student_img_feat.device
    teacher_img_feat = image_feats.to(device)
    teacher_text_feat = text_feats.to(device)

    # forward the language model and gather features
    student_text_feat = teacher_text_feat

    # compute the loss between img teacher logits and img student logits
    if args.use_udist_loss:
        teacher_img_logits, _ = models['create_logits'](teacher_img_feat, teacher_img_feat)
        student_img_logits, _ = models['create_logits'](student_img_feat, student_img_feat)
        udist_loss = 2 * compute_kl_loss(kl_loss, teacher_img_logits, student_img_logits, args.t_udist)
    else:
        udist_loss = 0
    # img_losses.update(img_loss.item() if img_loss != 0 else 0)

    if args.use_pvl_loss:
        teacher_img_logits, _ = models['create_logits'](teacher_img_feat, teacher_img_feat)
        student_img_logits, _ = models['create_logits'](student_img_feat, teacher_img_feat)
        pvl_loss = 2 * compute_kl_loss(kl_loss, teacher_img_logits, student_img_logits, args.t_pvl)
    else:
        pvl_loss = 0

    # compute the loss between teacher (image-text) and student (image-text)
    teacher_img_text_logits, teacher_text_img_logits = models['create_logits'](teacher_img_feat, teacher_text_feat)
    student_img_text_logits, student_text_img_logits = models['create_logits'](student_img_feat, student_text_feat)
    img_text_loss = compute_kl_loss(kl_loss, teacher_img_text_logits, student_img_text_logits, args.t_vl)
    text_img_loss = compute_kl_loss(kl_loss, teacher_text_img_logits, student_text_img_logits, args.t_vl)
    vl_loss = img_text_loss + text_img_loss

    loss = args.w_vl_loss * vl_loss + args.w_pvl_loss * pvl_loss + args.w_udist_loss * udist_loss
    return loss, vl_loss, pvl_loss, udist_loss


def train(models, dataloaders, optimizers, schedulers, args, epoch, writer):
    losses = AverageMeter()
    vl_losses = AverageMeter()
    pvl_losses = AverageMeter()
    udist_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    text_train_enumerator = enumerate(dataloaders['text_train_dataloader'])
    models['student_img_model'].train()

    if args.local_rank == -1:
        kl_loss = nn.KLDivLoss(reduction="batchmean")
    else:
        kl_loss = nn.KLDivLoss(reduction="batchmean").cuda(args.local_rank)
    start_time = time.time()
    for idx, (images, image_feats) in enumerate(dataloaders["img_train_dataloader"]):
        text_idx, (texts, text_feats) = next(text_train_enumerator)
        if text_idx == len(dataloaders['text_train_dataloader']) - 1:
            text_train_enumerator = enumerate(dataloaders['text_train_dataloader'])

        data_time.update(time.time() - start_time)
        start_time = time.time()

        loss, vl_loss, pvl_loss, udist_loss = get_loss(models, images, image_feats, text_feats, args, kl_loss)

        zero_grad(optimizers)
        loss.backward()
        step(optimizers)
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
                epoch, idx, len(dataloaders['img_train_dataloader']), batch_time=batch_time,  data_time=data_time,
                loss=losses, vl_loss=vl_losses, pvl_loss=pvl_losses, udist_loss=udist_losses), flush=True)
            writer.add_scalar('Train/total_loss', losses.avg, idx + args.n_step_per_epoch * epoch)
            writer.add_scalar('Train/vl_loss', vl_losses.avg, idx + args.n_step_per_epoch * epoch)
            writer.add_scalar('Train/pvl_loss', pvl_losses.avg, idx + args.n_step_per_epoch * epoch)
            writer.add_scalar('Train/udist_loss', udist_losses.avg, idx + args.n_step_per_epoch * epoch)

    return batch_time.avg, losses.avg, vl_losses.avg, pvl_losses.avg, udist_losses.avg

