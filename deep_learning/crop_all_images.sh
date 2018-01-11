seg=$1
mkdir images_cropped
for dir in images/*
do
    basepath=$(basename $dir)
    echo $basepath
    indir="images/"
    outdir="images_cropped/"
    mkdir $outdir
    FILES=$indir/*.jpg
    for f in $FILES
    do
	filename=$(basename $f .jpg)
	echo "Processing $f file..."
	convert $f -crop $seg"x"$seg"@" +repage +adjoin -resize 600x600 $outdir"/"$filename"_"$seg"x"$seg"_"%03d".jpg"
    done
done

