import numpy as np
import os.path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import json
import re
import collections
import cv2

def parse_labelmap(path):
  # item {
  #   id: 1
  #   name: 'rema'
  # }
  #
  # to
  #
  # {1: {'id': 1, 'name': u'rema'}}
  with open(path) as f:
    labels = json.loads("[%s]" % re.sub(" *([^ :]*):", '"\\1":', f.read()).replace("'", '"').replace("item {\n", "{").replace("\n}", "}").replace("\n", ","))
    return {label['id']: label for label in labels}

PATH_TO_CKPT = os.path.join(os.path.dirname(__file__), 'checkpoints/saved_model_4947.pb/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(os.path.dirname(__file__), 'label_map.pbtxt')


def merge_boxes(boxes,
                classes,
                scores,
                category_index,
                instance_masks=None,
                keypoints=None,
                max_boxes_to_draw=20,
                min_score_thresh=.5,
                line_thickness=4):
  """Extracted from tensorflow.research.object_detection.utils.visualization_utils.visualize_boxes_and_labels_on_image_array"""
  box_to_class_map = collections.defaultdict(str)
  box_to_class_name_map = collections.defaultdict(str)
  box_to_score_map = collections.defaultdict(float)
  box_to_instance_masks_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
  for i in range(min(max_boxes_to_draw, boxes.shape[0])):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if scores is None:
        box_to_class_map[box] = -1
      else:
        if classes[i] in category_index.keys():
          class_name = category_index[classes[i]]['name']
        else:
          class_name = 'N/A'
        box_to_class_map[box] = classes[i]
        box_to_class_name_map[box] = class_name
        box_to_score_map[box] = 100*scores[i]
        
  return {box: {"class": cls,
                "class_name": box_to_class_name_map.get(box),
                "score": box_to_score_map.get(box),
                "mask": box_to_instance_masks_map.get(box),
                "keypoints": keypoints and box_to_keypoints_map.get(box)}
          for box, cls in box_to_class_map.iteritems()}


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = parse_labelmap(PATH_TO_LABELS)

  
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def find_labels(image):  
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
      detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # This is how the model was trained, so we have to use the same byte order
      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      height, width = image.shape[:2]
      
      return {(int(bbox[0]*width), int(bbox[1]*height), int(bbox[2]*width), int(bbox[3]*height)): attrs
              for bbox, attrs in 
              merge_boxes(
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index).iteritems()}

if __name__ == "__main__":
  res = find_labels(sys.argv[1])
  import pdb
  pdb.set_trace()
