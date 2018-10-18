import os.path
import argparse
import matplotlib.pyplot as plt
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
    parser.add_argument('-o', '--out_dir', type=str,
                        help='name of the directory where detected person images will be saved to.')
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
for img_file in arg.images:
    print("processing [{}] ...".format(img_file))
    img = read_image(img_file)
    model = YOLOv3(pretrained_model='voc0712')
    model.score_thresh = arg.threshold
    bboxes, labels, scores = model.predict([img])
    bboxes, labels, scores = pickup_person(bboxes[0], labels[0], scores[0])
    img = img.transpose(1, 2, 0).astype(np.uint8)  # convert image format from CHW to HWC
    for idx, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int)
        person_img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        if arg.out_dir is not None:
            img_file = os.path.basename(img_file)
            out_base, out_ext = os.path.splitext(img_file)
            out_file = "{}/{}-{}{}".format(arg.out_dir, out_base, idx, out_ext)
            person_img_bgr = person_img[:, :, ::-1]
            cv2.imwrite(out_file, person_img_bgr)
            print("ourput a person to {} ...".format(out_file))
        else:
            plt.imshow(person_img)
            plt.show()
            input("please input [return] key.")
