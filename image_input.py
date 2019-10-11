#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 17:56:55 2019

@author: amvar
"""

#usage - python3 image_input.py --image car_high.jpg --classes classes.txt --weights yolov3.weights --config yolov3.cfg

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 19:53:10 2019

@author: amvar
"""
import os
import cv2
import glob
import argparse
import numpy as np
#import os
#import color_spectrum
import time

if not os.path.exists('image_result/'):
    os.mkdir('image_result/')

#import color_histogram_feature_extraction
#import knn_classifier
#import color_kmeans
#import color_classification
# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image',
                help = 'path to input image')
ap.add_argument('-c', '--config', default='yolov3-tiny.cfg',
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights',default='yolov3-tiny_900.weights',
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes',default='classes.txt',
                help = 'path to text file containing class names')
args = ap.parse_args()

count=0
for i in glob.glob('image/*'):
    print(i)
    image = cv2.imread(i)
    image=cv2.resize(image, (608,608))

    #sd = ShapeDetector()
    #cl = ColorLabeler()


    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    #scale = .5

    # read class names from text file
    classes = None
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # generate different colors for different classes 
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    start1 = time.time()

    COLORS[:,0]=255
    COLORS[:,1]=255
    COLORS[:,2]=0

    # read pre-trained model and config file
    net = cv2.dnn.readNet(args.weights, args.config)

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (608,608), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)



    # read input weights:
    #net = cv2.dnn.readNet(args.weights, args.config)

    # prepare i/p to run through deep neural network
    #blob = cv2.dnn.blobFromImage(image, scale, (Width,Height), (0,0,0), True, crop=False)

    #net.setInput(blob)
    end1 = time.time()
    net_time1=(end1 - start1)
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds to load".format(net_time1))



    # function to get the output layer names 
    # in the architecture
    def get_output_layers(net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,255), 6)

        cv2.putText(img, label, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)




    start2 = time.time()

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = .2
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])





    arr=np.array([[0,0,0,0,0]])
    arr=np.ndarray.astype(arr,dtype='str',casting='unsafe')

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # go through the detections remaining
    # after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        #print(x,y,w,h)
        
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        crop = image[int(y):int(y+h),int(x):int(x+w)]
        #for c in crop:
            #if w>100 and h>100:
                #cv2.imwrite('cropped/{}.jpg'.format(i),crop)

        #color_detect(image,class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))


        end2 = time.time()
        net_time2=(end2 - start2)
        
        # appending the info to an array
        arr=np.append(arr,[[str(classes[class_ids[i]]), class_ids[i],confidences[i],round(x+w),round(y+h)]],axis=0)

    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds to process an image".format(net_time2))
        
    #np.savetxt("images/output6.txt", arr, delimiter=",",fmt='%s',header='hello')   

    # display output image    
    #cv2.imshow("object detection", image)

    # To get the color of the cropped objects
    #cropped = color_classification.cropped_images('cropped/')

    # to get the color spectrum of all the cropped objects
    #color_spec = color_spectrum.color_spec('cropped/')


    # wait until any key is pressed
    # cv2.waitKey()
        
     # save output image to disk
    cv2.imwrite("image_result/{}.jpg".format(count), image)
    print('saved')
    count+=1

    #files = glob.glob('cropped/*.jpg')
#for f in files:
#    os.remove(f)


# release resources
#cv2.destroyAllWindows()
  
#print(arr)
