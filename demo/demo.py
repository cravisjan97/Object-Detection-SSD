import os
import math
import random
import time
import numpy as np
import tensorflow as tf
import cv2


from tensorflow.contrib import slim

# slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import sys
sys.path.append('../')


from preprocessing import ssd_vgg_preprocessing
from utility import visualization
from nets.ssd import g_ssd_model
import nets.np_methods as np_methods


# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)


# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.

predictions, localisations, _, _ = g_ssd_model.get_model(image_4d)

# Restore SSD model.
ckpt_filename = tf.train.latest_checkpoint('/home/sundaram/SSD_tensorflow_VOC-master/models_finetune/')

isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = g_ssd_model.ssd_anchors_all_layers()


# Main image processing routine.
def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
    
    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes



image_path = '/data/datasets/VOC07test/VOC2007/JPEGImages/'

i=0
'''
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/000593.jpg'  #two cars
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/001086.jpg'  #Two person and one bottle
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/009075.jpg'#many people, not very good
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/009957.jpg'  #horse and person
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/005575.jpg'   #plane
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/004865.jpg'  #horse
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/003552.jpg'  #car
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/003431.jpg'  #one person and one bottle
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/002808.jpg'  #motor and car
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/001672.jpg'  #train
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/001195.jpg'  #dog
image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/001302.jpg'
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/000568.jpg' #eagle
'''

lines = [line.rstrip('\n') for line in open('/data/datasets/VOC07test/VOC2007/ImageSets/Main/test.txt')]
print('Saving Images...')
start = time.time()
for l in lines:
	image_name = image_path+l+'.jpg'

	img = mpimg.imread(image_name)
	rclasses, rscores, rbboxes =  process_image(img)

	visualization.plt_bboxes(img, rclasses, rscores, rbboxes,l)
	#fig.savefig('/home/sundaram/SSD_tensorflow_VOC-master/demo/test_images_map_59/'+l+'.jpg')
elapsed = time.time()
print('Time Taken:%f' % (elapsed-start))
#visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
#image_name = '/data/datasets/VOC07test/VOC2007/JPEGImages/001302.jpg'
#img = mpimg.imread(image_name)
#rclasses, rscores, rbboxes =  process_image(img)
#visualization.plt_bboxes(img, rclasses, rscores, rbboxes)

#plt.savefig('myfig')
#plt.show()


