import numpy as np
import cv2
import imutils
from imutils import contours
import matplotlib.patches as patches
import pytesseract
import zbar
import sys
import json
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage.interpolation
import os
import traceback
import copy
import skimage.transform

# Tools for bbox handling

def boxDist(r1, r2):
    """Distance from center of right edge of r1 to center of left edge of r2 divided by their (max) height."""
    return np.sqrt((r1[0] + r1[2] - r2[0])**2 + 
                   (r1[1] + r1[3] / 2. - (r2[1] + r2[3] / 2.))**2) / float(max(r1[3], r2[3]))

def boxDiff(r1, r2):
    """'Unsigned' height ratio between boxes"""
    if r1[3] < r2[3]:
        return r1[3] / float(r2[3])
    else:
        return r2[3] / float(r1[3])

def bbox2pos(b):
    b = list(b)
    b[2] += b[0]
    b[3] += b[1]
    return b

def pos2bbox(p):
    p = list(p)
    p[2] -= p[0]
    p[3] -= p[1]
    return tuple(p)

def boxOverlap(r1, r2):
    r1 = bbox2pos(r1)
    r2 = bbox2pos(r2)
    x_overlap = max(0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    y_overlap = max(0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    return x_overlap * y_overlap
    
def imageLines(boxes, overlap_req = 0.3):
    lines = []
    for ridx, r in sorted(enumerate(boxes), lambda a, b: cmp(a[1][0], b[1][0])):
        overlaps = [lidx for lidx, l in enumerate(lines)
                    if boxOverlap(boxes[l[-1]], r) > r[2] * r[3] * overlap_req]
        if overlaps:
            lidx = overlaps[0]
            #print "%s added to %s after %s with due to overlap" % (ridx, lidx, lines[lidx][-1])
            lines[lidx].append(ridx)
        else:
            boxdists = sorted(((boxDist(boxes[l[-1]], r),
                                boxDiff(boxes[l[-1]], r),
                                lidx)
                               for lidx, l in enumerate(lines)),
                              lambda a, b: cmp(a[0], b[0]))
            boxdists = [(dist, diff, lidx)
                        for (dist, diff, lidx) in boxdists
                        if dist < 1 and diff > 0.75]
            if boxdists:
                dist, diff, lidx = boxdists[0]
                #print "%s added to %s after %s with distance %s and diff %s" % (ridx, lidx, lines[lidx][-1], dist, diff)
                lines[lidx].append(ridx)
            else:
                #print "%s added to new line %s" % (ridx, len(lines))
                lines.append([ridx])
    return lines

def lineBoxes(lines, boxes):
    lineboxes = []
    for line in lines:
        pos = bbox2pos(boxes[line[0]])
        for bidx in line[1:]:
            p = bbox2pos(boxes[bidx])
            if p[0] < pos[0]:
                pos[0] = p[0]
            if p[1] < pos[1]:
                pos[1] = p[1]
            if p[2] > pos[2]:
                pos[2] = p[2]
            if p[3] > pos[3]:
                pos[3] = p[3]
        lineboxes.append(pos2bbox(pos))
    return lineboxes

def flattenLines(lines, linegroups):
    return [[ridx for lidx in group for ridx in lines[lidx]]
            for group in linegroups]

def imageLinesRecursive(boxes):
    lines = imageLines(boxes)
    linelen = len(lines)
    oldLinelen = linelen + 1
    while linelen < oldLinelen:
        lineboxes = lineBoxes(lines, boxes)
        linegroups = imageLines(lineboxes)
        lines = flattenLines(lines,linegroups)
        oldLinelen = linelen
        linelen = len(lines)
    return lines
                 
    
# Tools for image handling

def lineKernels(size, *angles):
    kern = np.zeros((size, size), dtype="uint8")
    for i in xrange(0, size):
        kern[i, int(size/2)] = 1
    for angle in angles:
        yield scipy.ndimage.interpolation.rotate(kern, angle)

def imageGrads(img):
    grads = []
    for dx, dy in ((1, 0), (0, 1)):
        grad = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=dx, dy=dy,ksize=9)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        if maxVal == minVal: maxVal = minVal + 1
        grad = (255 * ((grad - minVal) / (maxVal - minVal)))
        grad = grad.astype("uint8")
        grads.append(grad)
    return grads[0] + grads[1]

def hierarchy2dicts(hier):
    children = {}
    parents = {}
    for self, (next, prev, firstchild, parent) in enumerate(hier[0]):
        if parent not in children: children[parent] = []
        children[parent].append(self)
        parents[self] = parent
    return children, parents

def convexity(cnt):
    """How convex is a contour, as a fraction between
    0. (convex hull infinitely larger than contour) and 1. (entirely convex)"""
    cnta = cv2.contourArea(cnt)
    if cnta == 0: return 0
    hull = cv2.convexHull(cnt)
    hulla = cv2.contourArea(hull)
    return cnta / hulla

def borderSides(idx, parents, children):
    if parents[idx] != -1:
        return parents[idx], idx
    elif children[idx]:
        return idx, children[idx][0]
    else:
        return idx, idx

def imageBordersHough(grad, size=0.3):
    size = int(min(*grad.shape) * size)

    edges = cv2.adaptiveThreshold(grad, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,-2)

    lines = skimage.transform.probabilistic_hough_line(
        edges, threshold=size, line_length=size,
        line_gap=0)

    borders = np.zeros(grad.shape, dtype="uint8")
    if lines is not None:
        for idx, line in enumerate(lines):
            (x1,y1),(x2,y2) = line
            cv2.line(borders, (x1, y1), (x2, y2), 255, 8)
    return borders, []

def imageObjgrad(grad, borders):
    bingrad = cv2.adaptiveThreshold(grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,-2)
    return cv2.threshold(bingrad - borders,
                         0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def imageContours(objgrad):
    cnts = cv2.findContours(objgrad.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return [cnt for cnt in cnts
            if cv2.contourArea(cnt) > 25]

def imageBoxes(objgrad):
    return [cv2.boundingRect(c) for c in imageContours(objgrad)]

def cut_image(img, bbox, hmargin=0, vmargin=0):
    (x, y, w, h) = bbox
    x -= hmargin
    y -= vmargin
    w += 2*hmargin
    h += 2*vmargin
    return img[y:y + h, x:x + w]
    
def is_barcode(img, cutoff=0.2):
    h, w = img.shape

    grad = imageGrads(img)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        
    vkern = cv2.getStructuringElement(cv2.MORPH_RECT, (1,int(h/2.)))
    hkern = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w/2.), 1))

    vimg = cv2.erode(thresh, vkern)
    himg = cv2.erode(thresh, hkern)

    vimg = np.bitwise_or.reduce(vimg, axis=0)
    himg = np.bitwise_or.reduce(himg, axis=1)

    vbarcodiness = vimg.sum() / 255. / len(vimg) > cutoff
    hbarcodiness = himg.sum() / 255. / len(himg) > cutoff
    
    #print "vimg", len(vimg), vimg.sum() / 255., vimg.sum() / 255. / len(vimg)
    #print "himg", len(himg), himg.sum() / 255., himg.sum() / 255. / len(himg)
    
    if (   (vbarcodiness and not hbarcodiness)
        or (not vbarcodiness and hbarcodiness)):
        return True
    else:
        return False

def imageTexts(img, lineboxes, hmargin=0, vmargin=0):
    linetexts = []
    for b in lineboxes:
        (x, y, w, h) = b
        txt = ""
        if w >= 10 and h >= 10:
            roi = cut_image(img, b, hmargin=hmargin, vmargin=vmargin)
            if roi.shape[0] and roi.shape[1]:
                if is_barcode(roi):
                    scanner = zbar.Scanner()
                    results = scanner.scan(roi)
                    if results:
                        txt = "%s: %s" % (results[0].type, results[0].data)
                else:
                    txt = pytesseract.image_to_string(Image.fromarray(roi))
        linetexts.append(txt)
    return linetexts

def mergeLineBoxesAndtexts(lineboxes, linetexts):
    return [(txt, b)
            for txt, b in zip(linetexts, lineboxes)
            if txt]

def normalizeImage(img, borders=None):
    p = img
    if borders is not None:
        p = p[borders == 0]
    s = p.std()
    m = np.median(p)
    return (((img - (m - s)) / (2 * s)).clip(0., 1.) * 255).astype("uint8")


def readLabelImage(image, noOcr = False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad = imageGrads(gray)
    borders, borderCnts = imageBordersHough(grad)
    objgrad = imageObjgrad(grad, borders)
    cnts = imageContours(objgrad)
    boxes = imageBoxes(objgrad)
    lines = imageLinesRecursive(boxes)
    lineboxes = lineBoxes(lines, boxes)
    
    if not noOcr:
        linetexts = imageTexts(gray, lineboxes, vmargin=2, hmargin=2)
    else:
        linetexts = ["" for box in lineboxes]
    return image, lineboxes, linetexts, borders, objgrad

def readLabel(filename, *arg, **kw):
    return readLabelImage(cv2.imread(filename), *arg, **kw)

# Tools for visualizing intermediate results

def drawLines(ax, lines, boxes):
    cmap = plt.get_cmap("prism")
    for idx, l in enumerate(lines):
        color = cmap(idx / float(len(lines)))
        for bidx in l:
            b = boxes[bidx]
            ax.add_patch(patches.Rectangle((b[0],b[1]),b[2],b[3], fill=False, edgecolor=color))

def drawCountours(ax, cnts, color="green"):
    for c in cnts:
        ax.plot(c[:,0,0], c[:,0,1], color=color)
        
def drawLineParts(ax, lines, boxes):
    for idx, l in enumerate(lines):
        for bidx in l:
            b = boxes[bidx]
            ax.add_patch(patches.Rectangle((b[0],b[1]),b[2],b[3], fill=False, edgecolor="red"))

def drawLineBoxes(ax, lineboxes):
    cmap = plt.get_cmap("prism")
    for b in lineboxes:
        ax.add_patch(patches.Rectangle((b[0],b[1]),b[2],b[3], fill=False, edgecolor="blue"))

def drawLineBoxesAndText(ax, lineboxes, linetexts, hmargin=0, vmargin=0):
    cmap = plt.get_cmap("prism")
    for lidx, (b, txt) in enumerate(zip(lineboxes, linetexts)):
        # if not txt: continue
        if b[2] < 10 or b[3] < 10: continue
        ax.add_patch(patches.Rectangle((b[0]-hmargin,b[1]-vmargin),b[2]+2*hmargin,b[3]+2*vmargin, fill=False, edgecolor="blue"))
        ax.text(b[0], b[1], "%s: %s" % (lidx, txt),
                horizontalalignment='left',
                verticalalignment='bottom',
                fontsize=12,
                color="red")

def renderLineBoxesAndText(out, lineboxes, linetexts, hmargin=0, vmargin=0):
    cmap = plt.get_cmap("prism")
    for lidx, (b, txt) in enumerate(zip(lineboxes, linetexts)):
        # if not txt: continue
        if b[2] < 10 or b[3] < 10: continue
        p = bbox2pos(b)
        cv2.rectangle(out, (p[0]-hmargin,p[1]-vmargin), (p[2]+2*hmargin,p[3]+2*vmargin), (0,0,255),2)
        cv2.putText(out, ("%s: %s" % (lidx, txt)).encode("utf-8"), (p[0], p[1]), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 2)

def makedir_for_file(path):
    path, file = os.path.split(path)
    if not os.path.exists(path):
        os.makedirs(path)

def transparent0_cmap(name="jet"):
    cmap = copy.deepcopy(plt.get_cmap(name))
    cmap._init()
    cmap._lut[0,-1] = 0.0
    return cmap
        
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
            image, lineboxes, linetexts, borders, objgrad = readLabel(path, **kws)

            print json.dumps({"path": path, "texts": mergeLineBoxesAndtexts(lineboxes, linetexts)}, ensure_ascii=False).encode("utf-8")
                        
            basepath, name = os.path.split(path)

            output = "%s/labeled/%s" % (basepath, name)
            makedir_for_file(output)
            out = image.copy()
            renderLineBoxesAndText(out, lineboxes, linetexts, vmargin=2, hmargin=2)
            cv2.imwrite(output, out)

            output = "%s/objgrad/%s" % (basepath, name)            
            makedir_for_file(output)
            fig, ax = plt.subplots(1,1, figsize=(15,10))
            ax.imshow(image)
            ax.imshow(objgrad, cmap=transparent0_cmap("autumn"))
            ax.imshow(borders, cmap=transparent0_cmap("winter"))
            fig.savefig(output)
            
        except Exception, e:
            print json.dumps({"path": path, "error": str(e), "traceback": traceback.format_exc()})
