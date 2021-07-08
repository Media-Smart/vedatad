# RGB Stream Is Enough for Temporal Action Detection
## Introduction
An accurate yet efficient one-stage temporal action detector based on single RGB stream

![Architecture](https://github.com/Media-Smart/vedatad/blob/main/configs/trainval/daotad/img/Architecture.png)

## Data Preparation

### THUMOS14

a. Download datasets & Create annotations

Follow the [official instructions](https://github.com/Media-Smart/vedatad/tree/main/tools/data/thumos14) on vedatad

b. Extract frames

```bash
cd ${vedatad_root}/data/thumos14
${vedatad_root}/tools/data/extract_frames.sh videos/val frames/val -vf fps=25 -s 128x128 %05d.jpg
${vedatad_root}/tools/data/extract_frames.sh videos/test frames/test -vf fps=25 -s 128x128 %05d.jpg
```

## Train

Follow the [official instructions](https://github.com/Media-Smart/vedatad#train) on vedatad

## Test

Follow the [official instructions](https://github.com/Media-Smart/vedatad#test) on vedatad

## Results and Weights
### THUMOS14
|  Model |  Batch Size | GPUs | AP50 | Config | Download |
|:------:|:-----------------------:|:----:|:----:|:------:|:--------:|
| daotad_i3d_r50_e700_thumos14_rgb | 16 | 4 | 0.538 | [config](https://github.com/Media-Smart/vedatad/blob/main/configs/trainval/daotad/daotad_i3d_r50_e700_thumos14_rgb.py) | model weights on [Google Drive](https://drive.google.com/drive/folders/151ueiYJrkL4YtnUktVDQoJ4tir9WdvKB) |
