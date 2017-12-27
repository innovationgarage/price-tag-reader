pos="positives/"
neg="negatives/"

#make list
cd $pos
ls -d "$PWD"/* > ../positives.lst
cd ..
cd $neg
ls -d "$PWD"/* > ../negatives.lst
cd ..

#grayscale images
cd $pos
mogrify -path . -type Grayscale *.jpg
cd ..
cd $neg
mogrify -type Grayscale *.jpg
mogrify -type Grayscale *.png
cd ..


