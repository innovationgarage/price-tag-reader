supermarket=$1
feature=$2
path=$feature"_"$supermarket

pos="positives/"
neg="negatives/"
numpos=200
numneg=5000
numstages=20
memuse=2000


## Pre-processing
mkdir $pos
mkdir $neg
mkdir $path

#grayscale images
echo $path
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

# ##Train the cascade
if [ $feature=='LBP' ];
then
    #LBP cascade
    opencv_traincascade -data LBP_$supermarket -vec positives.vec -bg negatives.lst -numStages $numstages -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos $numpos -numNeg $numneg -w 120 -h 80 -mode ALL -precalcValBufSize $memuse -precalcIdxBufSize $memuse -featureType LBP
else
    #Haar cascade
    opencv_traincascade -data $supermarket -vec positives.vec -bg negatives.lst -numStages $numstages -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos $numpos -numNeg $numneg -w 120 -h 80 -mode ALL -precalcValBufSize $memuse -precalcIdxBufSize $memuse
fi    
