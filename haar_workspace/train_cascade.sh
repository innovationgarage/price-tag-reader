supermarket=$1

pos="positives/"
neg="negatives/"
numpos=20
memuse=2000

## Pre-processing
mkdir $pos
mkdir $neg
mkdir $supermarket
mkdir LBP_$supermarket

#grayscale images
echo $supermarket
echo "grayscaling positive images"
mogrify -path $pos -type Grayscale ../haar_classifier/new_inputsets/$supermarket/*.jpg
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
perl createtrainsamples.pl positives.lst negatives.lst samples $numpos "opencv_createsamples -bgcolor 0 -bgthresh 80 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 120 -h 80 -nosym"
python mergevec.py -v samples/ -o positives.vec

##Train the cascade
#opencv_traincascade -data classifier -vec positives.vec -bg negatives.lst -numPos $numpos -numStages 20 -w 120 -h 80
#Haar cascade
#opencv_traincascade -data haar $supermarket -vec positives.vec -bg negatives.lst -numPos $numpos -precalcValBufSize $memuse -precalcIdxBufSize $memuse -minHitRate 0.995 -maxFalseAlarmRate 0.5 -weightTrimRate 0.95 -numStages 20 -w 120 -h 80

#LBP cascade
opencv_traincascade -data lbp LBP_$supermarket -vec positives.vec -bg negatives.lst -numPos $numpos -precalcValBufSize $memuse -precalcIdxBufSize $memuse -minHitRate 0.995 -maxFalseAlarmRate 0.5 -weightTrimRate 0.95 -numStages 20 -w 120 -h 80
