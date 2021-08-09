# OpenMMLab Competition 2021

This repository is used to participate in the [OpenMMLab Open Source Eco Challenge](https://openmmlab.com/competitions/algorithm-2021). The open source repository will host unofficial implementations of the [RepVGG](https://arxiv.org/abs/2101.03697) based on the [OpenMMLab](https://openmmlab.com/) open source framework [MMClassification](https://github.com/open-mmlab/mmclassification).

## Requirements

- MIM >= 0.1.2
- mmcv >= 1.3.8
- MMClassification >= 0.13

You can refer to [MIM](https://github.com/open-mmlab/mim) to install them. The approximate installation commands are as follows:

```shell
# Create virtual environment.
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# Install PyTorch. You can change the PyTorch and CUDA version as needed.
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch

# Install MIM.
pip install openmim

# Install MMCV and MMClassification through MIM.
mim install mmcv-full
mim install mmcls
```

## File Structure

```shell
.
├── .git
├── .gitignore
├── .pre-commit-config.yaml
├── README.md
├── RepVGG
│   └── repvgg.py
├── configs
│   ├── _base_
│   │   ├── datasets
│   │   │   └── imagenet_bs64.py
│   │   ├── default_runtime.py
│   │   ├── models
│   │   │   └── repvggA0.py
│   │   └── schedules
│   │       └── imagenet_bs256.py
│   └── repvgg
│       ├── README.md
│       ├── deploy
│       │   ├── repvggB2g4_64x4_imagenet_deploy.py
│       │   └── repvggB3_64x4_imagenet_deploy.py
│       ├── repvggA0_b64x4_imagenet.py
│       ├── repvggA1_64x4_imagenet.py
│       ├── repvggA2_64x4_imagenet.py
│       ├── repvggB0_64x4_imagenet.py
│       ├── repvggB1_64x4_imagenet.py
│       ├── repvggB1g2_64x4_imagenet.py
│       ├── repvggB1g4_64x4_imagenet.py
│       ├── repvggB2_64x4_imagenet.py
│       ├── repvggB2g2_64x4_imagenet.py
│       ├── repvggB2g4_64x4_imagenet.py
│       ├── repvggB3_64x4_imagenet.py
│       ├── repvggB3g2_64x4_imagenet.py
│       ├── repvggB3g4_64x4_imagenet.py
│       └── repvggB3g4_64x4_imagenet_autoaugment_mixup_warmup_coslr.py
├── setup.cfg
└── tools
    └── convert_repvggblock_param_to_deploy.py
```

## Models & Accuracy

See [README.md](https://github.com/zhangrui-wolf/openmmlab-competition-2021/blob/main/configs/repvgg/README.md) for information about the model, including the model's parameter sizes, GFLOPs, and accuracy. You can also get the download path of the log file and checkpoint file.

## Usages

### Train & Test

There are two ways to run this project: one is to copy the file to the MMClassification workspace and run the program directly based on the commands in MMClassification; the other is to run the project directly based on MIM.

#### MMClassification

1. Copy files to MMClassification workspace.

   ```shell
   cd openmmlab-competition-2021/
   cp RepVGG/repvgg.py ${mmcls_workspace}/mmcls/models/backbones/
   cp configs/_base_/models/repvggA0.py ${mmcls_workspace}/configs/_base_/models/
   cp -r configs/repvgg/ ${mmcls_workspace}/configs/
   sed -i "1c custom_imports = dict(imports=['mmcls.models.backbones.repvgg'], allow_failed_imports=False)" ${mmcls_workspace}/configs/_base_/models/repvggA0.py
   ```

   - **Note:**  ${mmcls_workspace} refers to the root directory of the MMClassification project.

2. Take RepVGG_B3 as an example:

   Before using it, please download and process the dataset and set the path in the configuration file.

   **Note:** If you are using a single GPU for training or testing, it is recommended that you first run "pip install -e ." .

   - Train

     ```shell
     # Single GPU
     python tools/train.py configs/repvgg/repvggB3_64x4_imagenet.py --work_dir ${YOUR_WORK_DIR}

     # Multiple GPUs
     ./tools/dist_train.sh configs/repvgg/repvggB3_64x4_imagenet.py ${GPU_NUM} --work_dir ${YOUR_WORK_DIR}
     ```

   - Test

     ```shell
     # Single GPU
     python tools/test.py configs/repvgg/repvggB3_64x4_imagenet.py ${CHECKPOINT_FILE} [--metrics ${METRICS}] [--out ${RESULT_FILE}]

     # Multiple GPUs
     ./tools/dist_test.sh configs/repvgg/repvggB3_64x4_imagenet.py ${CHECKPOINT_FILE} ${GPU_NUM} [--metrics ${METRICS}] [--out ${RESULT_FILE}]
     ```

#### MIM

1. Enter the RepVGG folder:

   ```shell
   cd openmmlab-competition-2021/
   ```

2. Take RepVGG_B3 as an example:

   Before using it, please download and process the dataset and set the path in the configuration file.

   **Note:** If you are using a single GPU for training or testing, it is recommended that you first run "mim install ." .

   - Train

     ```shell
     # Single GPU
     PYTHONPATH=$PWD:$PYTHONPATH mim train mmcls configs/repvgg/repvggB3_64x4_imagenet.py --work-dir ${YOUR_WORK_DIR} --gpus 1

     # Multiple GPUs
     PYTHONPATH=$PWD:$PYTHONPATH mim train mmcls configs/repvgg/repvggB3_64x4_imagenet.py --gpus ${GPU_NUM} --work-dir ${YOUR_WORK_DIR} --launcher pytorch
     ```

   - Test

     ```shell
     # Single GPU
     PYTHONPATH=$PWD:$PYTHONPATH mim test mmcls configs/repvgg/repvggB3_64x4_imagenet.py --checkpoint ${CHECKPOINT_FILE} --gpus 1 [--metrics ${METRICS}] [--out ${RESULT_FILE}]

     # Multiple GPUs
     PYTHONPATH=$PWD:$PYTHONPATH mim test mmcls configs/repvgg/repvggB3_64x4_imagenet.py --checkpoint ${CHECKPOINT_FILE} --gpus ${GPU_NUM} --launcher pytorch [--metrics ${METRICS}] [--out ${RESULT_FILE}]
     ```

### Reparameterize

Reparameterization is the most important feature of RepVGG networks. By reparameterization, the branching can be reduced and the speed of the model can be improved while keeping the accuracy of the model unchanged.

```shell
PYTHONPATH=$PWD:$PYTHONPATH python ./tools/convert_repvggblock_param_to_deploy.py ${config_path} ${checkpoint_path} ${save_path} [--device ${device}]
```

- `config_path`: The path of a model config file.
- `checkpoint_path`: The path of a model checkpoint file.
- `save_path`: Save path of the converted parameters.
- `device`: Which device the model is loaded to, optional: cpu, cuda.

Take the reparameterization of RepVGG_B3 as an example. Suppose the path to the checkpoints file is `checkpoints/RepVGG/B3/RepVGG_B3.pth`.

```shell
PYTHONPATH=$PWD:$PYTHONPATH python ./tools/convert_repvggblock_param_to_deploy.py  configs/repvgg/repvggB3_64x4_imagenet.py checkpoints/RepVGG/B3/RepVGG_B3.pth checkpoints/RepVGG/B3/RepVGG_B3_deploy.pth
```

Inference using the reparameterized parameters:

```shell
PYTHONPATH=$PWD:$PYTHONPATH mim test mmcls configs/repvgg/deploy/repvggB3_64x4_imagenet_deploy.py --checkpoint checkpoints/RepVGG/B3/RepVGG_B3_deploy.pth --gpus ${GPU_NUM} --launcher pytorch --metrics [--metrics ${METRICS}] [--out ${RESULT_FILE}]
```

## Citation

If you find this project useful in your research, please consider cite:

```latex
@inproceedings{ding2021repvgg,
  title={Repvgg: Making vgg-style convnets great again},
  author={Ding, Xiaohan and Zhang, Xiangyu and Ma, Ningning and Han, Jungong and Ding, Guiguang and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13733--13742},
  year={2021}
}

@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```
