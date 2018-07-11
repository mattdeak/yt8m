#!/bin/bash

mkdir -p full/frame full/video sample/frame sample/video
types=( video frame )
sets=( train validate test )

read -p "Subsample or full data? [S/F] " size
if [[ $size =~ ^[Ss]$ ]]
then
    for type in "${types[@]}"
    do
        cd sample/$type
        for set in "${sets[@]}"
        do
            curl -N data.yt8m.org/download.py | tac | tac | shard=1,100 partition=2/$type/$set mirror=us python3
        done
    done
elif [[ $size =~ ^[Ff]$ ]]
then
    for type in "${types[@]}"
    do
        cd full/$type
        for set in "${sets[@]}"
        do
            curl -N data.yt8m.org/download.py | tac | tac | partition=2/$type/$set mirror=us python3
        done
        cd ../..
    done
else
    echo "idk man"
fi
