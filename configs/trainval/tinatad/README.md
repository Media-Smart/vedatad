## Data Preparation

### Thumos14

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
