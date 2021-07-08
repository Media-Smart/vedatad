# RGB Stream Is Enough for Temporal Action Detection
## Introduction
An accurate yet efficient one-stage temporal action detector based on single RGB stream

![Architecture](https://github.com/Media-Smart/vedatad/blob/main/configs/trainval/daotad/img/Architecture.png)

## Data Preparation

### THUMOS14

a. Create data directory
```bash
mkdir -p ${vedatad_root}/data/thumos14
cd ${vedatad_root}/data/thumos14
```

b. Download raw annotations and video data

```bash
${vedatad_root}/tools/data/thumos14/download.sh
```

c. Create json format annotations for validation and test

```bash
python ${vedatad_root}/tools/data/thumos14/txt2json.py --anno_root annotations --video_root videos --mode val
python ${vedatad_root}/tools/data/thumos14/txt2json.py --anno_root annotations --video_root videos --mode test
```

d. Extract frames

```bash
${vedatad_root}/tools/data/extract_frames.sh videos/val frames/val -vf fps=25 -s 128x128 %05d.jpg
${vedatad_root}/tools/data/extract_frames.sh videos/test frames/test -vf fps=25 -s 128x128 %05d.jpg
```

## Train

Follow the official instructions on [vedatad](https://github.com/Media-Smart/vedatad#train)

## Test

Follow the official instructions on [vedatad](https://github.com/Media-Smart/vedatad#test)

## Results and Weights
### THUMOS14
|  Model |  Batch Size & GPU nums  | AP50 | Config | Download |
|:------:|:-----------------------:|:----:|:------:|:--------:|
| daotad_i3d_r50_e700_thumos14_rgb | 16 / 4 | 0.538 | [config](https://github.com/Media-Smart/vedatad/blob/main/configs/trainval/daotad/daotad_i3d_r50_e700_thumos14_rgb.py) | model weights on [Google Drive](https://drive.google.com/drive/folders/151ueiYJrkL4YtnUktVDQoJ4tir9WdvKB) and [Baidu Drive]()|
