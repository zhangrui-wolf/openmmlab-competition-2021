# Repvgg: Making vgg-style convnets great again

## Introduction

```latex
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13733--13742},
  year={2021}
}
```

## Pretrain model

Pre-training parameters of RepVGG are being converted into parameters that MMCls can use.

## Results and models

### ImageNet

|         Model          |             Params(M)             |            Flops(G)             |            Top-1 (%)            | Top-5 (%) |                            Config                            |                           Download                           |
| :--------------------: | :-------------------------------: | :-----------------------------: | :-----------------------------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      RepVGG-B2g4       |  61.76 (train) \| 55.78 (deploy)  | 12.63 (train) \| 11.34 (deploy) |              76.83              |   93.50   | [config (train)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/repvggB2g4_64x4_imagenet.py) \| [config (deploy)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/deploy/repvggB2g4_64x4_imagenet_deploy.py) | [model](https://drive.google.com/file/d/1rb4pEAA1LxWJp_CHoxic8oA-9byrEMkf/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qo9HdVs3dAhVDu5DbpKxTfRvsja7AH5J/view?usp=sharing) |
|       RepVGG-B3        | 123.09 (train) \| 110.96 (deploy) | 29.17 (train) \| 26.22 (deploy) | 77.88 (train) \| 77.87 (deploy) |   93.99   | [config (train)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/repvggB3_64x4_imagenet.py) \| [config (deploy)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/deploy/repvggB3_64x4_imagenet_deploy.py) | [model](https://drive.google.com/file/d/12n8iVZ9ayXrVZAib4OeHbDU2vg1c4k1u/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qo9HdVs3dAhVDu5DbpKxTfRvsja7AH5J/view?usp=sharing) |
| RepVGG-B3g4 (training) |                                   |                                 |                                 |           | [config (train)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/repvggB3g4_64x4_imagenet_autoaugment_mixup_warmup_coslr.py) |                                                              |
