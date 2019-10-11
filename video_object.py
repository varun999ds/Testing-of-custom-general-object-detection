
import cv2
import argparse
import numpy as np
import os
import glob



Config="yolov3-food100.cfg"
Weights="yolov3-food100_2000.weights"
Classes="classes.txt"
conf_threshold = 0.6
nms_threshold = 0.3
video_path='video/4.mp4'

if not os.path.exists('video_result/'):
    os.mkdir('video_result/')

# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0,0,0), 4)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
 

p=[]



# Load names classes
classes = None
with open(Classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
#print(classes)

#Generate color for each class randomly
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# COLORS[:,0]=0
# COLORS[:,1]=255
# COLORS[:,2]=255


# Define network from configuration file and load the weights from the given weights file
net = cv2.dnn.readNet(Weights,Config)


# Define video capture for default cam
cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print(cap)
count=0

while (count<total):
    class_ids = []
    confidences = []
    boxes = []
 
    count+=1               
    hasframe, image = cap.read()
    # if count%5 !=0:
    #     continue

    #print(count,"frame count")
    #image=cv2.resize(image, (416,416)) 

    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (608,608), [0,0,0], True, crop=False)
    net.setInput(blob)
    outs = net.forward(getOutputsNames(net))
    #print(len(outs)) 

    
    # In case of tiny YOLOv3 we have 2 output(outs) \
    #from 2 different scales [3 bounding box per each scale]
    # For normal normal YOLOv3 we have 3 output(outs)\
    # from 3 different scales [3 bounding box per each scale]
    
    # For tiny YOLOv3, the first output will be 507x6 = 13x13x18
    # 18=3*(4+1+1) 4 boundingbox offsets, 1 objectness prediction, and 1 class score.
    # and the second output will be = 2028x6=26x26x18 (18=3*6) 
    
    for out in outs: 
        #print(out.shape)
        for detection in out:   
        #each detection  has the form like this 
        #[center_x ,center_y, width, height, obj_score, class_1_score, class_2_score ..]
            scores = detection[5:]#classes scores starts from index 5
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    
    # apply  non-maximum suppression algorithm on the bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i] 
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        
        #if class_ids[i]==32:
        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        
            # if count<=7:
            #     p.append((round(x),round(y)))
            # else:
            #     p.pop(0)
            #     p.append((round(x),round(y)))
        
            # c=0    
            # for i in p:
            #     c=c+1
            #     cv2.circle(image, i, c,color=(0,255,0), thickness=-1, lineType=8, shift=0)
        #arr1=np.array([[str(classes[class_ids[i]]), class_ids[i],confidences[i],round(x),round(y)]])
        #arr=np.append(arr,[[str(classes[class_ids[i]]), class_ids[i],confidences[i],round(x),round(y)]],axis=0)
        #print(arr1)
    
    
    # Put efficiency information.
   # t, _ = net.getPerfProfile()
    #label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    #cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
    
    #cv2.imshow(window_title, image)


    cv2.imwrite('video_result/{}.jpg'.format(count),image)
    
    #name='image(%d).jpg' % count
    #cv2.imwrite(name,image)
    
cap.release()
#cv2.destroyAllWindows()      
        
# p[0][1]
# for i in p:
#     print(i[0])
