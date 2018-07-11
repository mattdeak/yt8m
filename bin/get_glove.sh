#!/bin/bash
cd ../data; cd data; # can't be sure which one so do both, one will fail
mkdir -p glove
if [ ! -z `ls glove/*` ]
then
    echo "Glove already downloaded. Exiting."
    exit
fi

wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
mv glove.840B.300d.txt glove.txt
split -d -l 10000 glove.txt
mv x* glove/
rm glove.txt
