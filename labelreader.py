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
            if p[3] > pos[0]:
                pos[3] = p[3]
        lineboxes.append(pos2bbox(pos))
    return lineboxes

    
# Tools for image handling
    
def imageGrads(img):
    grads = []
    for dx, dy in ((1, 0), (0, 1)):
        grad = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=dx, dy=dy,ksize=9)
        grad = np.absolute(grad)
        (minVal, maxVal) = (np.min(grad), np.max(grad))
        grad = (255 * ((grad - minVal) / (maxVal - minVal)))
        grad = grad.astype("uint8")
        grads.append(grad)
    return grads[0] + grads[1]

def imageBorders(grad, width=5, length=0.25):
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (width,width))
    borders = [None, None]
    for idx, kern in enumerate((cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(length * grad.shape[1]))),
                                cv2.getStructuringElement(cv2.MORPH_RECT, (int(length * grad.shape[0]),1)))):
        borders[idx] = cv2.erode(grad, kern)
        borders[idx] = cv2.threshold(borders[idx], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        borders[idx] = cv2.dilate(borders[idx], rectKern)
        borders[idx] = cv2.dilate(borders[idx], kern)
    return borders[0] + borders[1]
    
def imageObjgrad(grad, borders):
    bingrad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return cv2.threshold(bingrad - borders,
                         0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def imageContours(objgrad):
    cnts = cv2.findContours(objgrad.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return cnts[0] if imutils.is_cv2() else cnts[1]

def imageBoxes(objgrad):
    return [cv2.boundingRect(c) for c in imageContours(objgrad)]

def imageLines(boxes):
    lines = []
    for ridx, r in sorted(enumerate(boxes), lambda a, b: cmp(a[1][0], b[1][0])):
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

def readLabel(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad = imageGrads(gray)
    objgrad = imageObjgrad(grad)
    boxes = imageBoxes(objgrad)
    lines = imageLines(boxes)
    lineboxes = lineBoxes(lines, boxes)
    linetexts = imageTexts(gray, lineboxes)
    return [(txt, b)
            for txt, b in zip(linetexts, lineboxes)
            if txt]

# Tools for visualizing intermediate results

def drawLines(ax, lines, boxes):
    cmap = plt.get_cmap("prism")
    for idx, l in enumerate(lines):
        color = cmap(idx / float(len(lines)))
        for bidx in l:
            b = boxes[bidx]
            ax.add_patch(patches.Rectangle((b[0],b[1]),b[2],b[3], fill=False, edgecolor=color))

def drawCountours(ax, cnts):
    for c in cnts:
        ax.plot(c[:,0,0], c[:,0,1], color="green")
        
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

if __name__ == "__main__":
    print json.dumps(readLabel(sys.argv[1]))
