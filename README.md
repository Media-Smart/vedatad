## Introduction
vedatad is a single stage temporal action detection toolbox based on PyTorch.

## Features

- **Modular Design**

  We decompose detector into four parts: data pipeline, model, postprocessing and criterion which make it easy to convert PyTorch model into TensorRT engine and deploy it on NVIDIA devices such as Tesla V100, Jetson Nano and Jetson AGX Xavier, etc.

- **Support of several popular single stage detector**

  The toolbox supports several single stage detector out of the box, *e.g.* tinatad, etc.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher
- ffmpeg

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- PyTorch 1.8.0
- Python 3.8.5
- ffmpeg 4.3.11

### Install vedatad

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedatad python=3.8.5 -y
conda activate vedatad
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedatad repository.

```shell
git clone https://github.com/Media-Smart/vedatad.git
cd vedatad
vedatad_root=${PWD}
```

d. Install vedatad.

```shell
pip install -r requirements/build.txt
pip install -v -e .
```

## Data preparation

Please follow specified algorithm in `config/trainval` to prepare data, for example, see detail in `configs/trainval/tinatad`.

## Train

a. Config

Modify some configuration accordingly in the config file like `configs/trainval/tinatad/tinatad.py`

b. Train
```shell
tools/dist_trainval.sh configs/trainval/tinatad/tinatad.py "0,1"
```

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/trainval/tinatad/tinatad.py`

b. Test
```shell
CUDA_VISIBLE_DEVICES="0" python tools/test.py configs/trainval/tinatad/tinatad.py weight_path
```

## Contact

This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [vedadet](https://github.com/Media-Smart/vedadet), thanks to [Media-Smart](https://github.com/Media-Smart).
