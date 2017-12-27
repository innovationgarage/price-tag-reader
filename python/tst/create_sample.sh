#opencv_createsamples -vec positive1511343179504.vec -img positives/1511343179504.jpg -bg negatives.txt -num 100 -w 120 -h 80 -bgcolor 0 -bgthresh 80 -nosym
opencv_createsamples -vec positives.vec -img positives/* -bg negatives.lst -w 120 -h 80 -bgcolor 0 -bgthresh 80 -nosym
