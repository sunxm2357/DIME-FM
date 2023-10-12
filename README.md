# DIME-FM
Implementation of "DIME-FM: DIstilling Multimodal and Efficient Foundation Models" (ICCV 2023)

## Abstract
Large **V**ision-**L**anguage **F**oundation **M**odels (VLFM), such as CLIP, ALIGN and Florence, are trained on large-scale datasets of image-caption pairs and achieve superior transferability and robustness on downstream tasks, but they are difficult to use in many practical applications due to their large size, high latency and fixed architectures. Unfortunately, recent work shows training a small custom VLFM for resource-limited applications is currently very difficult using public and smaller-scale data. In this paper, we introduce a new distillation mechanism (**DIME-FM**) that allows us to transfer the knowledge contained in large VLFMs to smaller, customized foundation models using a relatively small amount of inexpensive, unpaired images and sentences. **We transfer the knowledge from the pre-trained CLIP-ViT- L/14 model to a ViT-B/32 model, with only 40M public im- ages and 28.4M unpaired public sentences.** The resulting model “Distill-ViT-B/32” rivals the CLIP-ViT-B/32 model pre-trained on its private WiT dataset (400M image-text pairs): Distill-ViT-B/32 achieves similar results in terms of zero-shot and linear-probing performance on both Ima- geNet and the ELEVATER (20 image classification tasks) benchmarks. It also displays comparable robustness when evaluated on five datasets with natural distribution shifts from ImageNet. 

Links: [Arxiv](https://arxiv.org/abs/2303.18232)/[Project Page](https://cs-people.bu.edu/sunxm/DIME-FM/)/Poster/Slides

Welcome to cite our work if you find it is helpful to your research.
```
@article{sun2023dime,
  title={DIME-FM: DIstilling Multimodal and Efficient Foundation Models},
  author={Sun, Ximeng and Zhang, Pengchuan and Zhang, Peizhao and Shah, Hardik and Saenko, Kate and Xia, Xide},
  journal={arXiv preprint arXiv:2303.18232},
  year={2023}
}
```

## Release TODO List
- [x] Checkpoints
- [x] Evaluation code
- [x] Training code (Expected by the end of Oct)

## Checkpoints
| Model | Image Training Set | Text Training Set |ZS on IN-1K | ZS on ELEVATER | LP on ELEVATER | Robustness | Download
| :----: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViT-B/32 | IN-21K + GCC-15M + YFCC-14M | Filtered Roberta NLP Corpus | 66.5% | 56.4% | 79.2% | 50.2% | [ckpt](https://drive.google.com/drive/folders/1P_SY5kJ2CSbXKvGzEWYnnw1c3ufBupYJ?usp=sharing)
| ViT-B/32 | IN-21K + GCC-15M + YFCC-14M | IN-21K Prompts + GCC-15M + YFCC-14M + Downstream Tasks' Prompts | 66.1% | 57.7% |  79.4% | - | [ckpt](https://drive.google.com/drive/folders/1u9bW_J2azACwN4r8SVy88WREw-wIwWom?usp=sharing)

## Evaluation
Our evaluation is based on [ELEVATER](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC) benchmark (Please refer to [README](Evaluation/README.md)  in `Evaluation` folder). We extend `ELEVATER` benchmark to include the ImageNet Variants as the Robustness Evaluation. The download links of these datasets can be found  [HERE](Evaluation/README.md). 

## Training
We provide the training code in [Training](Training) folder.

### Environment
```angular2html
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm 
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
pip install transformers
pip install yacs
```

### Prepare the Data
For faster dataloading speed, we pack all data in `tsv` files according to [this repo](https://github.com/microsoft/scene_graph_benchmark/tree/main/tools/mini_tsv). We extract the features of image and text using CLIP-ViT-L/14 models. All features are also stored in `tsv` form.

We provide examples of image, text, image features, text features.

|     Data      |                                                                                          Download Link                                                                                          |
|:-------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|     Image     | [tsv](https://drive.google.com/file/d/1OBR7N3YA3BLEks0vOCNX8rK8DjkNo7qw/view?usp=drive_link) / [lineidx](https://drive.google.com/file/d/1fyBeZOgbwWgO-iM0rkgw0_hmnB8QDtT9/view?usp=drive_link) |
|     Text      | [tsv](https://drive.google.com/file/d/1SNvG4xbrP_WiEex-X1VF-4H3ib75Ub_0/view?usp=drive_link) / [lineidx](https://drive.google.com/file/d/1EXLa1BpJJEJRvkP4wG2nEfMFaf3i0SBQ/view?usp=drive_link) |
| Image Feature | [tsv](https://drive.google.com/file/d/1ljwZlQP56nV4RLOMYG9fIlLQkavjtmhb/view?usp=drive_link) / [lineidx](https://drive.google.com/file/d/1ifp2aMDEIxK1mjZVShpApEQngeAhjaOG/view?usp=drive_link) |
| Text Feature  |                                           [tsv](https://drive.google.com/file/d/1SkxBtzSDkwQ1QYqUyrzhSPKj2TvE-UOc/view?usp=drive_link) / [lineidx](https://drive.google.com/file/d/1CBjz6BnldtE69K4ihQmD5iwJOYiD2w3R/view?usp=drive_link)                                            |

### Training command
```angular2html
cd Training
python train_amp.py  --amp --dataroot <your_dataroot> --tsv_file_list  configs/datalists/cc3m/image.list \
configs/datalists/cc3m/image_feat.list configs/datalists/cc3m/text.list  configs/datalists/cc3m/text_feat.list \
--batch_size <your_batch_size> --use_pvl_loss 
```
Please refer to [Data Parallel Example](https://github.com/pytorch/examples/blob/main/distributed/ddp/README.md) to extend to multiple gpus and nodes.
