import numpy as np
import cv2
import sys
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import labelreader
import json
import traceback
import deep_learning.my_model_inference

def findLabelsHaar(img, cascade_path="OpenCV_cascades/HAAR_all_14/cascade.xml", scale_factor=1.3, min_neighbors=5, **kw):
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, scale_factor, min_neighbors)


def readLabelsImage(img, algorithm="tfobj", **kw):
    if algorithm=="tfobj":
        tags = deep_learning.my_model_inference.find_labels(img, **kw)
    else:
        tags = findLabelsHaar(img, cascade_path, scale_factor, min_neighbors, **kw)
    bbox = [bbox for bbox in tags][0]    
    return [(tuple(int(x) for x in bbox),) + labelreader.readLabelImage(labelreader.cut_image(img, bbox), **kw)
            for bbox in tags]

        
def readLabels(filename, *arg, **kw):
    return readLabelsImage(cv2.imread(filename), *arg, **kw)

if __name__ == "__main__":
    args = []
    kws = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            value = True
            arg = arg[2:]
            if '=' in arg:
                arg, value = arg.split("=", 1)
            kws[arg] = value
        else:
            args.append(arg)
                
    for path in sys.stdin:
        path = path[:-1]
        try:
            for bbox, image, lineboxes, linetexts, borders, objgrad in readLabels(path, **kws):
                print json.dumps({"path": path, "bbox": bbox, "texts": labelreader.mergeLineBoxesAndtexts(lineboxes, linetexts)}, ensure_ascii=False).encode("utf-8")            
        except Exception, e:
            raise
            print json.dumps({"path": path, "error": str(e), "traceback": traceback.format_exc()})
