import numpy as np
import cv2
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser()
parser.add_argument('--cascade', '-c', type=str, required=True, help='supermarket name')
parser.add_argument('--test', '-t', type=str, required=True, help='path to the test image')
parser.add_argument('--mode', '-m', type=str, default="detect", help='set to "face" or "eye# to use openCV pre-trained cascades')
args = vars(parser.parse_args())

if args['mode']=='detect':
    cascade_path = os.path.join(args['cascade'], 'cascade.xml')
elif args['mode']=='face':
    cascade_path = "/home/saghar/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
elif args['mode']=='eye':    
    cascade_path = "/home/saghar/opencv/data/haarcascades/haarcascade_eye.xml"

print cascade_path
cascade = cv2.CascadeClassifier(cascade_path)
img = cv2.imread(args['test'])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

tags = cascade.detectMultiScale(gray, 1.3, 5)
fig,ax = plt.subplots(1)
ax.imshow(img)
for (x,y,w,h) in tags:
    rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
#    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

#cv2.imshow('img', img)
#cv2.imwrite('detection.jpg', img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

plt.show()
