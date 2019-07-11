"""
Created July, 2019
@author: Hassan Yousuf & Nabeel Hussain 
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

image_size=200
images = []

# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

sess = tf.Session()
graph = tf.get_default_graph()
saver = tf.train.import_meta_graph('~/tf-realsense-gesture/.meta')
saver.restore(sess, tf.train.latest_checkpoint('./')) 
y_pred = graph.get_tensor_by_name("y_pred:0")
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((1, 7))

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
#depth scale is in meters that can be change according to the applications
depth_scale = 0.0030000000474974513

# removing the background of objects more than
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
        
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        ret1, bw_img1 = cv2.threshold(bg_removed,127,255,cv2.THRESH_BINARY_INV)
        ret2, bw_img2 = cv2.threshold(depth_colormap,127,255,cv2.THRESH_BINARY)
        
	bitmap=cv2.resize(bw_img1,(200,200),cv2.CV_16S,1)
	bitmap = bitmap.astype('uint8')
	newimg=cv2.multiply(bitmap,255)
        x_batch=newimg.reshape(1,image_size,image_size,3)
	feed_dict_testing = { x:x_batch, y_true: y_test_images}
	result=sess.run(y_pred, feed_dict=feed_dict_testing)
        images = np.hstack((newimg, newimg))

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)

	print("Up  :"+str(result[0][0]))
	print("Down :"+str(result[0][1]))
	print("Left  :"+str(result[0][2]))
	print("Right  :"+str(result[0][3]))
	print("Takeoff  :"+str(result[0][4]))
	print("Land  :"+str(result[0][5]))
	print("None  :"+str(result[0][6]))
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
 pipeline.stop()
