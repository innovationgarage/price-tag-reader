import numpy as np
import cv2
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_on_image(img_path, cascade_path, out_path, scale_factor, min_neighbors):
    cascade = cv2.CascadeClassifier(cascade_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tags = cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    for (x,y,w,h) in tags:
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.savefig(out_path)

parser = argparse.ArgumentParser()
parser.add_argument('--cascade', '-c', type=str, required=True, help='supermarket name')
parser.add_argument('--testdir', '-t', type=str, required=True, help='path to the dir containing test images')
parser.add_argument('--mode', '-m', type=str, default="detect", help='set to "face" or "eye# to use openCV pre-trained cascades')
parser.add_argument('--outdir', '-o', type=str, required=True, help='path to the output directory')
parser.add_argument('--scale factor', '-sf', type=str, default=1.3, help='detectMultiScale scale factor')
parser.add_argument('--min neighbors', '-mn', type=int, default=5, help='detectMultiScale min neighbors')

args = vars(parser.parse_args())

if args['mode']=='detect':
    cascade_path = os.path.join(args['cascade'], 'cascade.xml')
elif args['mode']=='face':
    cascade_path = "/home/saghar/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
elif args['mode']=='eye':    
    cascade_path = "/home/saghar/opencv/data/haarcascades/haarcascade_eye.xml"
print cascade_path

tst_img_path = args['testdir']
tst_imgs = [os.path.join(tst_img_path, x) for x in os.listdir(tst_img_path)]

out_img_path = args['outdir']
if not os.path.exists(out_img_path):
    try:
        os.makedirs(out_img_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
out_imgs = [os.path.join(out_img_path, x) for x in os.listdir(tst_img_path)]        

for i, im in enumerate(tst_imgs):
    print(im)
    test_on_image(im, cascade_path, out_imgs[i], float(args['scale factor']), args['min neighbors'])
                  
