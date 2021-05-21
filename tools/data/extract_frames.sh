#!/usr/bin/env bash

INPUT_DIR=$1
OUTPUT_DIR=$2

if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

for video in $(ls $INPUT_DIR)
do
    video_name=${video%.*}
    if [ ! -d $OUTPUT_DIR/$video_name ]; then
        mkdir $OUTPUT_DIR/$video_name
    fi
    ffmpeg -i $INPUT_DIR/$video ${@:3:$#-3} $OUTPUT_DIR/$video_name/${!#}
done
