from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from io import BytesIO
import json
import logging
import base64
import random
from typing import Callable, List, Tuple, Union
from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError
import torch.utils.data as data
from .prompt_engineering import prompt_engineering
from .tsv_file import TSVFile, CompositeTSVFile
import numpy as np
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class TSVImageDataset(data.Dataset):
    """
        This class is intended for encapsulating Image/Text pair data for contrastive learning described in
        the following paper,
        "Learning Transferable Visual Models From Natural Language Supervision" (a.k.a CLIP)
        V2: support image text pairs and supervised classification data
    """
    def __init__(self,
                 image_tsv_file: Union[str, List[str]],
                 image_feat_tsv_file: List[str],
                 transform: Callable = None):
        self.transform = transform
        self._chunk_sizes = None
        self.image_tsv_file_list = []
        self.image_feat_tsv_file_list = []

        if isinstance(image_tsv_file, str):
            # single tsv file
            if (
                    os.path.splitext(image_tsv_file)[1].lower() == '.tsv'
            ):
                self.image_tsv_file_list.append(image_tsv_file)
                self.image_tsv_file = TSVFile(
                    image_tsv_file, if_generate_lineidx=True
                )
            else:
                raise ValueError("Invalid input! Please check the tsv filenames.")
        elif isinstance(image_tsv_file, list):
            self.image_tsv_file_list = [
                img
                for img in image_tsv_file
            ]
            self.image_tsv_file = CompositeTSVFile(
                image_tsv_file,
            )
            self._chunk_sizes = self.image_tsv_file.get_chunk_size()
        else:
            raise ValueError("Invalid input! Please check the tsv filenames.")

        # prepare the feature files
        self.load_feat = True
        self.image_feat_tsv_file_list = [
                img_feat
                for img_feat in image_feat_tsv_file
            ]
        self.image_feat_tsv_file = CompositeTSVFile(
            image_feat_tsv_file,
        )

        img_feat_chunk_sizes = self.image_feat_tsv_file.chunk_sizes
        logging.info('num of image file: %d, num of image feat file: %d' % (len(self._chunk_sizes), len(img_feat_chunk_sizes)))

        assert len(self.image_tsv_file) == len(self.image_feat_tsv_file)

        self.len_image_tsv_file = len(self.image_tsv_file)

    def get_chunk_sizes(self):
        return self._chunk_sizes

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if index is None:
            import torch
            return torch.tensor([], dtype=torch.float32), \
                torch.tensor([], dtype=torch.int64), \
                torch.tensor([], dtype=torch.int64)

        # iterative over image.
        items_image = self.image_tsv_file[index]

        img_key, img = self._decode_image(items_image)

        if self.transform:
            img = self.transform(img)

        items_image_feat = self.image_feat_tsv_file[index]
        image_feat = self._decode_feat(items_image_feat)
        return img, image_feat

    def _decode_feat(self, items: Tuple[str, str]):
        key = items[0]
        try:
            feat = np.frombuffer(base64.b64decode(items[1]), np.float32)
        except:
            print(key)
            print(items[1])
            raise ValueError

        return torch.tensor(feat)

    def _decode_image(self, items: Tuple[str, str]):
        key = items[0]
        # image = Image.open(BytesIO(base64.b64decode(items[1]))).convert('RGB')

        try:
            image = Image.open(BytesIO(base64.b64decode(items[1])))
            return key, image.convert('RGB')
        except (ValueError, UnidentifiedImageError):
            logger.info('Failed in Loading Images')
            return 'sunxm_warn', Image.new('RGB', (256, 256), (int(0.48145466 * 255), int(0.4578275 * 255), int(0.40821073 * 255)))

    def __len__(self):
        return len(self.image_tsv_file)


class TSVTextDataset(data.Dataset):
    """
        This class is intended for encapsulating Image/Text pair data for contrastive learning described in
        the following paper,
        "Learning Transferable Visual Models From Natural Language Supervision" (a.k.a CLIP)
        V2: support image text pairs and supervised classification data
    """
    def __init__(self,
                 text_tsv_file: Union[str, List[str]],
                 text_feat_tsv_file: List[str],
                 tokenize: Callable = None,
                 context_length: int = 77,
                 num_captions: int = 1,
                 text_format: str = 'txt'):
        self.tokenize = tokenize
        self._chunk_sizes = None
        self.context_length = context_length
        self.num_captions = num_captions
        self.text_format = text_format
        self.text_tsv_file_list = []
        self.text_feat_tsv_file_list = []

        if isinstance(text_tsv_file, str):
            # single tsv file
            if (
                    os.path.splitext(text_tsv_file)[1].lower() == '.tsv'
            ):
                self.text_tsv_file_list.append(text_tsv_file)
                self.text_tsv_file = TSVFile(
                    text_tsv_file, if_generate_lineidx=True
                )
            else:
                raise ValueError("Invalid input! Please check the tsv filenames.")
        elif isinstance(text_tsv_file, list):
            self.text_tsv_file_list = [
                txt
                for txt in text_tsv_file
            ]
            self.text_tsv_file = CompositeTSVFile(
                text_tsv_file
            )
        else:
            raise ValueError("Invalid input! Please check the tsv filenames.")

        # prepare the feature files
        assert len(text_feat_tsv_file) > 0

        self.text_feat_tsv_file_list = [
            text_feat
            for text_feat in text_feat_tsv_file
        ]
        self.text_feat_tsv_file = CompositeTSVFile(
            text_feat_tsv_file
        )

        text_feat_chunk_sizes = self.text_feat_tsv_file.chunk_sizes
        text_chunk_sizes = self.text_tsv_file.chunk_sizes
        logging.info('num of text file: %d, num of text feat file: %d' % (
        len(text_chunk_sizes), len(text_feat_chunk_sizes)))
        assert len(self.text_tsv_file) == len(self.text_feat_tsv_file)

        self.len_text_tsv_file = len(self.text_tsv_file)

    def get_chunk_sizes(self):
        return self._chunk_sizes

    def __getitem__(self, index: Union[int, Tuple[int, int]]):
        if index is None:
            import torch
            return torch.tensor([], dtype=torch.float32), \
                torch.tensor([], dtype=torch.int64), \
                torch.tensor([], dtype=torch.int64)

        # iterative over image.
        items_text = self.text_tsv_file[index]

        _, txt = self._decode_text(items_text)

        tokens = self.tokenize(
            txt, padding='max_length', truncation=True, max_length=self.context_length,
            return_tensors='pt'
        ) if self.tokenize else txt

        tokens['input_ids'].squeeze_()
        tokens['attention_mask'].squeeze_()

        items_text_feat = self.text_feat_tsv_file[index]
        text_feat = self._decode_feat(items_text_feat)
        return tokens, text_feat

    def _decode_feat(self, items: Tuple[str, str]):
        key = items[0]
        try:
            feat = np.frombuffer(base64.b64decode(items[1]), np.float32)
        except:
            print(key)
            print(items[1])
            raise ValueError

        return torch.tensor(feat)

    def _decode_text(self, items: Tuple[str, Union[str, dict]]):
        key = items[0]
        text = ''

        if self.text_format != 'json':
            raise ValueError('Only support json format')

        try:
            js = json.loads(items[1])
        except json.decoder.JSONDecodeError:
            js = {'captions': items[1]}
        if isinstance(js, list):
            js = js[0]
        if 'captions' in js:
            captions = js['captions']
            if isinstance(captions, list):
                if self.num_captions == 1:
                    text = random.choice(captions)
                else:
                    text = captions
                    if len(captions) > self.num_captions:
                        text = captions[:self.num_captions]
            elif isinstance(captions, str):
                text = captions
            else:
                raise ValueError('captions should be str or list')
        elif 'caption' in js:
            captions = js['caption']
            if isinstance(captions, list):
                if self.num_captions == 1:
                    text = random.choice(captions)
                else:
                    text = captions
                    if len(captions) > self.num_captions:
                        text = captions[:self.num_captions]
            elif isinstance(captions, str):
                text = captions
            else:
                raise ValueError('captions should be str or list')
        elif 'tags' in js:
            text = prompt_engineering(js['tags'])
        elif 'task' in js and js['task'] == 'classification':
            text = prompt_engineering(js['class_name'])
        elif isinstance(js, str):
            text = js
        else:
            raise ValueError('js.keys are  ', js.keys())

        if "[unused0]" in text:
            text = text.replace("[unused0]", "person")

        return key, text

    def __len__(self):
        return len(self.text_tsv_file)