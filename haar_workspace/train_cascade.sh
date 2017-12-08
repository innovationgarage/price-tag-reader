supermarket=$1
feature=$2
path=$feature"_"$supermarket

pos="positives/"
neg="negatives/"
<<<<<<< HEAD
numpos=500
numneg=2000
numstages=7
=======
numpos=1000
numneg=1000
numstages=5
>>>>>>> 2ad83cc4d39efd787a5247d14fedd8d555042285
memuse=2000
w=120
h=80
size=$w"x"$h"^"
<<<<<<< HEAD
numThreads=8
=======
numThreads=7
>>>>>>> 2ad83cc4d39efd787a5247d14fedd8d555042285

## Pre-processing
mkdir $pos
mkdir $neg
mkdir $path

#grayscale images
echo $path
echo "grayscaling and resizing positive images"
mogrify -path $pos -type Grayscale -resize $size ../haar_classifier/new_inputsets/$supermarket/*.jpg
echo "grayscaling negative images"
mogrify -path $neg -type Grayscale ../haar_classifier/new_inputsets/negatives/*.jpg

#make list
cd $pos
ls -d "$PWD"/* > ../positives.lst
cd ..
cd $neg
ls -d "$PWD"/* > ../negatives.lst
cd ..

##Create/augment training images
perl createtrainsamples.pl positives.lst negatives.lst samples $numpos "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w "$w" -h "$h""
python mergevec.py -v samples/ -o positives.vec

##Train the cascade
<<<<<<< HEAD
opencv_traincascade -data $path -vec positives.vec -bg negatives.lst -numStages $numstages -numThreads $numThreads -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos $numpos -numNeg $numneg -w $w -h $h -precalcValBufSize $memuse -precalcIdxBufSize $memuse -featureType $feature #-mode ALL
=======
opencv_traincascade -data $path -vec positives.vec -bg negatives.lst -numStages $numstages -numThreads $numThreads  -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos $numpos -numNeg $numneg -w $w -h $h -precalcValBufSize $memuse -precalcIdxBufSize $memuse -featureType $feature -mode ALL
>>>>>>> 2ad83cc4d39efd787a5247d14fedd8d555042285
