import os
import argparse
import json
import glob
from collections import OrderedDict

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description='Generate annotation')
    parser.add_argument('--anno_root', type=str, help='annotations root dir')
    parser.add_argument('--video_root', type=str, help='videos root dir')
    parser.add_argument('--mode', choices=['val', 'test'])

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    anno_dir = os.path.join(args.anno_root, args.mode)
    video_dir = os.path.join(args.video_root, args.mode)

    database = OrderedDict()
    if args.mode == 'test':
        txt_files = glob.glob(os.path.join(anno_dir, '[B-Z]*.txt'))
    else:
        txt_files = glob.glob(os.path.join(anno_dir, '[A-Z]*.txt'))
    for txt_file in txt_files:
        class_name = os.path.split(txt_file)[-1].split('_')[0]
        with open(txt_file) as f:
            for line in f.readlines():
                video_name, start, end = line.strip().split()
                if video_name not in database:
                    database[video_name] = OrderedDict(annotations=[])
                database[video_name]['annotations'].append(
                    dict(segment=[float(start), float(end)], label=class_name))

    for video_name, video_info in database.items():
        video_file = os.path.join(video_dir, f'{video_name}.mp4')
        cap = cv2.VideoCapture(video_file)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_info['duration'] = duration
        video_info['resolution'] = f'{width}x{height}'
        video_info['subset'] = args.mode

        cap.release()

    out_file = os.path.join(args.anno_root, f'{args.mode}.json')
    out = OrderedDict(database=database, taxonomy=[], version='THUMOS14')
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=4)


if __name__ == '__main__':
    main()
