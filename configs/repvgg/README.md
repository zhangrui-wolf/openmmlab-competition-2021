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

|    Model    | Epochs |             Params(M)             |            Flops(G)             | Top-1 (%) | Top-5 (%) | Config | Download |
| :---------: | :----: | :-------------------------------: | :-----------------------------: | :-------: | :-------: | :----: | :------: |
|  RepVGG-A0  |  120   |   9.11（train) \| 8.31 (deploy)   |  1.52 (train) \| 1.36 (deploy)  |   71.43   |   90.16   |        |          |
|  RepVGG-A1  |  120   |  14.09 (train) \| 12.79 (deploy)  |  2.64 (train) \| 2.37 (deploy)  |   73.82   |   91.46   |        |          |
|  RepVGG-A2  |  120   |  28.21 (train) \| 25.5 (deploy)   |  5.7 (train)  \| 5.12 (deploy)  |   75.65   |   92.61   |        |          |
|  RepVGG-B0  |  120   |  15.82 (train) \| 14.34 (deploy)  |  3.42 (train) \| 3.06 (deploy)  |   74.42   |   92.09   |        |          |
|  RepVGG-B1  |  120   |  57.42 (train) \| 51.83 (deploy)  | 13.16 (train) \| 11.82 (deploy) |   77.72   |   93.88   |        |          |
| RepVGG-B1g2 |  120   |  45.78 (train) \| 41.36 (deploy)  |  9.82 (train) \| 8.82 (deploy)  |   77.30   |   93.56   |        |          |
| RepVGG-B1g4 |  120   |  39.97 (train) \| 36.13 (deploy)  |  8.15 (train) \| 7.32 (deploy)  |   76.69   |   93.36   |        |          |
|  RepVGG-B2  |  120   |  89.02 (train) \| 80.32 (deploy)  | 20.46 (train) \| 18.39 (deploy) |   78.10   |   94.07   |        |          |
| RepVGG-B2g4 |  120   |  61.76 (train) \| 55.78 (deploy)  | 12.63 (train) \| 11.34 (deploy) |   77.87   |   93.76   |        |          |
| RepVGG-B2g4 |  200   |  61.76 (train) \| 55.78 (deploy)  | 12.63 (train) \| 11.34 (deploy) |   78.87   |   94.44   |        |          |
|  RepVGG-B3  |  200   | 123.09 (train) \| 110.96 (deploy) | 29.17 (train) \| 26.22 (deploy) |   79.87   |   95.00   |        |          |
| RepVGG-B3g4 |  200   |  83.83 (train) \| 75.63 (deploy)  | 17.9 (train) \| 16.08 (deploy)  |   79.63   |   94.87   |        |          |

## Results and models

### ImageNet

|         Model          |             Params(M)             |            Flops(G)             |            Top-1 (%)            | Top-5 (%) |                            Config                            |                           Download                           |
| :--------------------: | :-------------------------------: | :-----------------------------: | :-----------------------------: | :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      RepVGG-B2g4       |  61.76 (train) \| 55.78 (deploy)  | 12.63 (train) \| 11.34 (deploy) |              76.83              |   93.50   | [config (train)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/repvggB2g4_64x4_imagenet.py) \| [config (deploy)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/deploy/repvggB2g4_64x4_imagenet_deploy.py) | [model](https://drive.google.com/file/d/1rb4pEAA1LxWJp_CHoxic8oA-9byrEMkf/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qo9HdVs3dAhVDu5DbpKxTfRvsja7AH5J/view?usp=sharing) |
|       RepVGG-B3        | 123.09 (train) \| 110.96 (deploy) | 29.17 (train) \| 26.22 (deploy) | 77.88 (train) \| 77.87 (deploy) |   93.99   | [config (train)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/repvggB3_64x4_imagenet.py) \| [config (deploy)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/deploy/repvggB3_64x4_imagenet_deploy.py) | [model](https://drive.google.com/file/d/12n8iVZ9ayXrVZAib4OeHbDU2vg1c4k1u/view?usp=sharing) \| [log](https://drive.google.com/file/d/1qo9HdVs3dAhVDu5DbpKxTfRvsja7AH5J/view?usp=sharing) |
| RepVGG-B3g4 (training) |  83.83 (train) \| 75.63 (deploy)  | 17.9 (train) \| 16.08 (deploy)  |                                 |           | [config (train)](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/repvggB3g4_64x4_imagenet_autoaugment_mixup_warmup_coslr.py) |                                                              |
