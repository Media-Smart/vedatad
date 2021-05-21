#!/usr/bin/env bash

wget http://crcv.ucf.edu/THUMOS14/Validation_set/TH14_Temporal_annotations_validation.zip -O annotations_val.zip
wget http://crcv.ucf.edu/THUMOS14/test_set/TH14_Temporal_annotations_test.zip -O annotations_test.zip

if [ ! -d "annotations" ]; then
    mkdir annotations
fi

for mode in "val" "test"
do
    if [ ! -d "annotations/$mode" ]; then
        mkdir annotations/$mode
    fi
    unzip -j annotations_$mode.zip -d annotations/$mode
    rm annotations_$mode.zip
done

wget https://storage.googleapis.com/thumos14_files/TH14_validation_set_mp4.zip -O videos_val.zip
wget https://storage.googleapis.com/thumos14_files/TH14_Test_set_mp4.zip -O videos_test.zip

if [ ! -d "videos" ]; then
    mkdir videos
fi

for mode in "val" "test"
do
    if [ ! -d "videos/$mode" ]; then
        mkdir videos/$mode
    fi
    unzip -P "THUMOS14_REGISTERED" -j videos_$mode.zip -d videos/$mode
    find videos/$mode/*.mp4 | grep -Ev $(echo $(find annotations/$mode/ -name "[A-Z]*.txt" | xargs cat | cut -d ' ' -f 1 | sort | uniq) | sed 's/ /|/g') | xargs rm
    rm videos_$mode.zip
done
