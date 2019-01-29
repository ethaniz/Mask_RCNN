# -*- coding:utf8 -*-
import cv2
import os
import sys
import tensorflow as tf
import numpy as np
import time
ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import custom

MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
custom_WEIGHTS_PATH = 'mask_rcnn_score_card_0009.h5'

config = custom.CustomConfig()
custom_DIR = os.path.join(ROOT_DIR, 'datasets')


image_path = sys.argv[1]


class InferenceConfig(config.__class__):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = '/cpu:0'
TEST_MODE = 'inference'

with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode='inference', model_dir=MODEL_DIR,
        config=config)

print("Loading weights", custom_WEIGHTS_PATH)
model.load_weights(custom_WEIGHTS_PATH, by_name=True)

image = cv2.imread(image_path)


start = time.time()
image_resized, window, scale, paddings, _ = utils.resize_image(
    image,
    min_dim=800,
    max_dim=1024,
    mode='square'
)

image_padded, window, scale, paddings, _ = utils.resize_image(
    image,
    min_dim=1280,
    max_dim=1280,
    mode='square'
)
end1 = time.time()
result = model.detect([image_resized], verbose=1)
end2 = time.time()

print("resize cost %f s ..." % (end1-start))
print("detect cost %f s ..." % (end2-end1))

import pdb
#pdb.set_trace()

r = result[0]
class_ids = r['class_ids']
label = r['rois']
masks = r['masks']

class_id_dict = {1: 'score', 2: 'time'}

for i in range(len(class_ids)):
    class_id = class_id_dict[class_ids[i]]
    p = r['masks'][:,:,i].flatten()
    p0 = np.where(p == True)[0][0]
    p1 = np.where(p == True)[0][-1]

    x0 = p0//image_resized.shape[0]
    y0 = p0%image_resized.shape[0]
    x1 = p1//image_resized.shape[0]
    y1 = p1%image_resized.shape[0]

    cv2.imwrite('%s.jpg' % class_id, image_resized[x0:x1, y0:y1, :])


#xy = r['rois'] * 1280 // 1024
#
#class_id_dict = {1: 'score', 2: 'time'}
#
#for i in range(len(xy)):
#    class_id = class_id_dict[class_ids[i]]
#    label_xy = xy[i]
#    ret = image_padded[label_xy[0]:label_xy[2], label_xy[1]:label_xy[3]]
#    cv2.imwrite('%s.jpg' % class_id, ret)


#i = 1
#for instance in xy:
#    delta = instance[3] - instance[1]
#    ret = image_padded[instance[0]:instance[2], instance[1]:instance[3]]
#    #pdb.set_trace()
#    cv2.imwrite('result_%i_a.jpg' % i, ret[:, :delta//2+1, ::-1])
#    cv2.imwrite('result_%i_b.jpg' % i, ret[:, delta//2+1:, ::-1])
#    i += 1






#delta_width = ret.shape[1] // 6



#cv2.imwrite('result_1.jpg', ret[:, delta_width:2*delta_width+1, ::-1])
#cv2.imwrite('result_2.jpg', ret[:, 3*delta_width+1:4*delta_width+1, ::-1])
#cv2.imwrite('result_3.jpg', ret[:, 4*delta_width+1:, ::-1])