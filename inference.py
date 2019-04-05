import argparse
import os
import math
import random
import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
from nets import ssd_vgg_300, ssd_vgg_512, np_methods
from preprocessing import ssd_vgg_preprocessing

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)                                                                                     # choose net here
#net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()                                                                             # choose net here!!!
#ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
ckpt_filename = '/root/train/model.ckpt-1666'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


# Main image processing routine.
def process_image(img, select_threshold=0.9999, nms_threshold=.05, net_shape=(512, 512)):
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-vid', '--video-input', dest="video_input", type=str,
                        default='/root/5.avi')
    parser.add_argument('-img', '--image_directory', dest='image_directory', type=str)
    args = parser.parse_args()

    if args.image_directory:
        print('Reading from image folder')
        images_mode = True
        images = iter(os.listdir(args.image_directory))
        cv2.namedWindow("Video")
        frame = None
        while True:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            if k == ord('n'):
                try:
                    fname = args.image_directory + next(images)
                    print(fname)
                    frame = cv2.imread(fname)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame is not None:
                        rclasses, rscores, rbboxes = process_image(frame)
                        print(len(rbboxes))
                        if len(rbboxes>0):
                            for point in rbboxes:
                                print(point)
                                cv2.rectangle(frame,
                                          (int((point[1]) * frame.shape[1]), int((point[0]) * frame.shape[0])),
                                          (int((point[3]) * frame.shape[1]), int((point[2]) * frame.shape[0])),
                                          (255, 0, 0), 3)
                except StopIteration:
                    break
    else:
        print('Reading from video.', args.video_input)
        video_capture = cv2.VideoCapture(args.video_input)
        ret = True
        frame = None
        while ret:
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            ret, frame = video_capture.read()
            if frame is not None:
                frame_dark = np.copy(frame)
                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])
                frame = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                rclasses, rscores, rbboxes = process_image(frame)
                if len(rbboxes>0):
                    for point in rbboxes:
#                        print(point)
                        roi = frame
                        cv2.rectangle(frame_dark,
                                  (int((point[1]) * frame.shape[1]), int((point[0]) * frame.shape[0])),
                                  (int((point[3]) * frame.shape[1]), int((point[2]) * frame.shape[0])),
                                  (255, 255, 255), 5)
                cv2.imshow("Vid", frame_dark)
        video_capture.stop()

