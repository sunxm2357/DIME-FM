from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pathlib
from os.path import basename

from timm.data import create_loader
import torch
import torch.utils.data
import torchvision.datasets as datasets
from pathlib import Path
from yacs.config import CfgNode as CN
from transformers import CLIPTokenizer
from transformers import AutoTokenizer
from .tsv import TSVImageDataset, TSVTextDataset
import torchvision.transforms as T

import os

logger = logging.getLogger(__name__)

INTERPOLATION_MODES = {
    'bilinear': T.InterpolationMode.BILINEAR,
    'bicubic': T.InterpolationMode.BICUBIC,
    'nearest': T.InterpolationMode.NEAREST,
}


def build_transform():
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )

    logger.info('=> use torchvision transform fro training')
    crop_size = 224
    interpolation = INTERPOLATION_MODES['bilinear']
    trans = [T.Resize(256, interpolation=INTERPOLATION_MODES['bicubic'])]

    trans.append(
        T.RandomResizedCrop(
            crop_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333),
            interpolation=interpolation))

    trans.append(T.RandomHorizontalFlip(0.5))
    trans.extend(
        [
            T.ToTensor(),
            normalize,
        ]
    )

    transforms = T.Compose(trans)
    return transforms


def build_tokenizer(tokenizer_type):
    tokenizer = None
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if tokenizer_type == 'clip':
        pretrained_tokenizer = 'openai/clip-vit-base-patch32'
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_tokenizer)
        tokenizer.add_special_tokens({'cls_token': tokenizer.eos_token})
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    return tokenizer


def build_dataset(transforms, tsv_list, dataroot):
    image_dataset, text_dataset = _build_unpaired_dataset(transforms, tsv_list, dataroot)

    return image_dataset, text_dataset


def _build_unpaired_dataset(transforms, tsv_list, dataroot):
    logger.info('transforms: {}'.format(transforms))

    # tokenobj = build_tokenizer(cfg['LANG_ENCODER']['TOKENIZER'])
    tokenobj = build_tokenizer('clip')

    assert len(tsv_list) > 0
    tsv_filenames = sorted(
        [
            os.path.join(dataroot, f)
            for f in tsv_list
        ]
    )

    image_tsv_files = [
        filename
        for filename in tsv_filenames
        if (
                'image-' in basename(filename)
                or 'images_' in basename(filename)
                or 'img' in basename(filename)
                or 'image_' in basename(filename)
                or '_image' in basename(filename)
                or '-image' in basename(filename)
        ) and 'feat' not in basename(filename)
    ]
    text_tsv_files = [
        filename
        for filename in tsv_filenames
        if (
                'text-' in basename(filename)
                or 'caption' in basename(filename)
                or 'text_' in basename(filename)
                or '_text' in basename(filename)
                or '-text' in basename(filename)
        ) and 'feat' not in basename(filename)
    ]

    logger.info(
        "=> found %d image tsv file(s) and %d text tsv file(s) to load." % (
        len(image_tsv_files), len(text_tsv_files))
    )

    image_feat_tsv_files = [
        filename
        for filename in tsv_filenames
        if (
                   'image-' in basename(filename)
                   or 'images_' in basename(filename)
                   or 'img' in basename(filename)
                   or 'image_' in basename(filename)
                   or '_image' in basename(filename)
                   or '-image' in basename(filename)
           ) and 'feat' in basename(filename)
    ]

    text_feat_tsv_files = [
        filename
        for filename in tsv_filenames
        if (
                   'text-' in basename(filename)
                   or 'caption' in basename(filename)
                   or 'text_' in basename(filename)
                   or '_text' in basename(filename)
                   or '-text' in basename(filename)
           ) and 'feat' in basename(filename)
    ]

    logger.info(
        "=> found %d image feature tsv file(s) and %d text feature tsv file(s) to load." % (
            len(image_feat_tsv_files), len(text_feat_tsv_files))
    )

    num_captions = 1
    text_format = 'json'

    image_dataset = TSVImageDataset(
        image_tsv_files,
        image_feat_tsv_files,
        transform=transforms,
    )

    text_dataset = TSVTextDataset(
         text_tsv_files, text_feat_tsv_files,
        tokenize=tokenobj,
        num_captions=num_captions,
        text_format=text_format,

    )

    logger.info(
        "=> %s image set size: %d", 'train', len(image_dataset)
    )

    logger.info(
        "=> %s text set size: %d", 'train', len(text_dataset)
    )

    return image_dataset, text_dataset


def build_dataloader(tsv_list, dataroot, batch_size_per_gpu, distributed=False, num_workers=8, pin_memory=True):
    transforms = build_transform()
    image_dataset, text_dataset = build_dataset(transforms, tsv_list, dataroot)

    batch_size_per_gpu = batch_size_per_gpu
    shuffle = True

    if distributed:
        image_sampler = torch.utils.data.distributed.DistributedSampler(
            image_dataset, shuffle=shuffle)
        text_sampler = torch.utils.data.distributed.DistributedSampler(
            text_dataset, shuffle=shuffle)
        shuffle = False
    else:
        image_sampler = None
        text_sampler = None

    image_data_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=image_sampler,
        drop_last=True
    )

    text_data_loader = torch.utils.data.DataLoader(
        text_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=text_sampler,
        drop_last=True
    )

    return image_data_loader, text_data_loader