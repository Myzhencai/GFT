import os, argparse

import tensorflow as tf
import cv2

import numpy as np
from pathlib import Path
import matplotlib
import time
matplotlib.use('Agg')
from freezeGraphClass import freezeGraph

# filepath = Path("C:/Users/soumi/Documents/test-Autoencoder").glob('*.jpg')
def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename,"rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        tf.import_graph_def(graph_def, name = "load")
        flops = tf.profiler.profile(graph,options=tf.profiler.ProfileOptionBuilder.float_operation())
        print(flops)
        print('FLOP = ', flops.total_float_ops)

    with tf.Session(graph = graph,config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #output = tf.get_default_graph().get_tensor_by_name("load/fcrelu:0")
        output = tf.get_default_graph().get_tensor_by_name("load/Inference/Output:0")
        inputPlaceholder = tf.get_default_graph().get_tensor_by_name("load/Input:0")
        # isTrain = tf.get_default_graph().get_tensor_by_name("load/isTrain:0")
        # isTrain = tf.get_default_graph().get_tensor_by_name("load/Inference/Output:0")
        global falseCount
        global trueCount
        finalAccuracyList = []
        closestList = []
        # model = Model(X, Y, learningRate)
        # prediction = model.prediction()
        image= '/home/gaofei/ooga/left_images/left_lei_15.jpg'
        image_reader = cv2.imread(image, 0)
        image_reader = cv2.resize(image_reader, (320, 240))
        normalised_image = image_reader.astype(np.float32)
        normalised_image = np.expand_dims(normalised_image, axis=2)
        normalised_image = np.expand_dims(normalised_image, axis=0)
        test_data = normalised_image
        test_mask2 = sess.run(output, feed_dict={inputPlaceholder: test_data})
        print("result:")
        image1 =image_reader-test_mask2[0,:,:,0]-test_mask2[0,:,:,1]-test_mask2[0,:,:,2]-test_mask2[0,:,:,3]-test_mask2[0,:,:,4]
        image2 =image_reader
        # print(test_mask2[0,:,:,0])
        cv2.imwrite('/home/gaofei/Semantic-Segmentation/test0.jpg',image1)
        cv2.imwrite('/home/gaofei/Semantic-Segmentation/test.jpg', image2)
        cv2.imwrite('/home/gaofei/Semantic-Segmentation/testwhich.jpg', image2-image1)
        for i in range(5):
            temimage = test_mask2[0,:,:,i]
            cv2.imwrite('/home/gaofei/Semantic-Segmentation/each{0}.jpg'.format(i), temimage)
        # filenames = [file for file in filepath]
        # previous = []
        # current = []
        # for file in filenames:
        #     filename, file_extension = os.path.splitext(str(file))
        #     name = str(os.path.basename(filename));
        #     print (name)
        #     test_data = []
        #     image = cv2.imread(str(file))
        #     imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #     clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        #     imagegray = clahe.apply(imagegray)
        #     # randomCircle(imagegray,5)
        #     normalised_image = imagegray
        #     normalised_image1 = normalised_image.astype(float)/255.0
        #     normalised_image1 = np.expand_dims(normalised_image1, axis=2)
        #     test_data.append(normalised_image1)
        #     start = time.perf_counter()
        #     test_mask2 = sess.run(output,feed_dict={inputPlaceholder:test_data, isTrain:False})
        #     elapsed = time.perf_counter() - start
        #     print ("elapsed",elapsed)
        #     print (test_mask2.shape)
        #     test_mask = test_mask2[0]
        #     '''
        #     current = test_mask
        #     previous.append(current)
        #     for p in previous:
        #         result = 1 - spatial.distance.cosine(p, current)
        #         print (result)
        #     '''
        #     output_mask = (((test_mask[:,:,0]).astype(np.float)))
        #     cv2.imshow('Output',output_mask)
        #     cv2.imshow('Input',imagegray)
        #     cv2.imwrite('image.jpg',output_mask)
        #     k = cv2.waitKey(0)
        #     if k == 27:
        #         break
outputTensorName =  "Inference/Output"
freezeGraph = freezeGraph('/home/gaofei/Semantic-Segmentation/model5/model_learningRate_0.01epochs_30/',outputTensorName)
freezeGraph.freeze_graph()
load_graph("/home/gaofei/Semantic-Segmentation/model5/model_learningRate_0.01epochs_30/frozen_model.pb")

