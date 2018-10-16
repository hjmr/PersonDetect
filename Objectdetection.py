n the rest of the tutorial, we assume that the `plt`
# is imported before every code snippet.
import matplotlib.pyplot as plt
import argparse
import cv2

import numpy as np

from chainercv.datasets import voc_bbox_label_names
from chainercv.links import YOLOv3
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox

LABEL_ID_PERSON = 14

def parse_arg():
    parser = argparse.ArgumentParser(description='Blah Blah')
    parser.add_argument('-t', '--threshold', type=float, default=0.6, help='set the score threshold.')
    parser.add_argument('images', type=str, nargs='+', help='a list of images.')
    return parser.parse_args()


def pickup_person(bbox_all, labels_all, scores_all):
    bboxes = []
    labels = []
    scores = []
    for bbox, label, score in zip(bbox_all, labels_all, scores_all):
        if label == LABEL_ID_PERSON:
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
    return np.array(bboxes), np.array(labels), np.array(scores)

arg = parse_arg()
plt.ion()
for img in arg.images:
    img = read_image(img)
    model = YOLOv3(pretrained_model='voc0712')
    model.score_thresh = arg.threshold
    bboxes, labels, scores = model.predict([img])
    bboxes, labels, scores = pickup_person(bboxes[0], labels[0], scores[0])
    vis_bbox(img, bboxes, labels, scores, label_names=voc_bbox_label_names)
    plt.draw()
    key = input("please input [return] key.")
    plt.close()


for (x,y,w,h) in arg.images:
    print("[x,y] = %d,%d [w,h] = %d,%d" %(x, y, w, h))
    cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), thickness=10)
plt.show()
cv2.imwrite("../output/out.jpg",img)

for (x,y,w,h) in arg.images:
    face_img = origin_img[y:y+h, x:x+w]
    filename = "face_" + str(x) + "-" + str(y) + ".jpg"
    cv2.imwrite(filename, face_img)

