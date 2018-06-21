#!/bin/bash

read -p "Subsample or full data? [S/F] " size
if [[ $size =~ ^[Ss]$ ]]
then
    sampling='shard=1,100'
    cd subsample/
elif [[ $size =~ ^[Ff]$ ]]
then
    echo "This one"
    sampling=''
    cd full/
else
    sampling='wut'
fi

cd video

trap 'exit' INT
curl -N data.yt8m.org/download.py | tac | tac | $sampling partition=2/video/train mirror=us python3
curl -N data.yt8m.org/download.py | tac | tac | $sampling partition=2/video/validate mirror=us python3
curl -N data.yt8m.org/download.py | tac | tac | $sampling partition=2/video/test mirror=us python3

# Frame-level
cd ..
cd frame
curl -N data.yt8m.org/download.py | tac | tac | $sampling partition=2/frame/train mirror=us python3
curl -N data.yt8m.org/download.py | tac | tac | $sampling partition=2/frame/validate mirror=us python3
curl -N data.yt8m.org/download.py | tac | tac | $sampling partition=2/frame/test mirror=us python3
