#!/usr/bin/env python
# coding: utf-8
import cv2
import numpy as np
import os
import sys
import yaml
from yaml.loader import SafeLoader
from src.exception import ObjectDetectionException


class YOLO_Pred():
    try :
        def __init__(self, onnx_model, data_yaml):
            # Load the YAML
            with open(data_yaml,"r") as f:
                data_yaml = yaml.load(f,Loader=SafeLoader)

            self.labels = data_yaml['names']
            self.nc = data_yaml['nc']

            # Load the yolo model
            self.yolo = cv2.dnn.readNetFromONNX(onnx_model) #----check
            self.yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        def predictions(self,image):

            row, column, d = image.shape
            ## STEP1 : CONVERT IMAGE INTO SQUARE IMAGE(ARRAY)
            max_rc = max(row,column)
            input_image = np.zeros((max_rc,max_rc,3), dtype = np.uint8) # 3 is for 3d array
            input_image[0:row ,0:column] = image

            # STEP2 : GET PREDICTION FROM SQUARE ARRAY
            INPUT_WH_YOLO = 640
            blob = cv2.dnn.blobFromImage(input_image,
                                        1/255,
                                        (INPUT_WH_YOLO,INPUT_WH_YOLO),
                                        swapRB=True, 
                                        crop=False)
            self.yolo.setInput(blob)
            #detection or prediction from YOLO model
            preds = self.yolo.forward() 

            # Non maximum supression
            # Step1- Filter detection based on confidence score=0.4 and probablity score-0.25
            detections = preds[0]
            boxes = []
            confidences = []
            classes = []

            # width and height of the image (input_image)
            image_w, image_h = input_image.shape[:2]
            x_factor = image_w/INPUT_WH_YOLO
            y_factor = image_h/INPUT_WH_YOLO

            for i in range(len(detections)):
                row = detections[i]
                confidence = row[4] #confidence of detection of an object

                # if confidence >0.4 then only select the bounding box.
                if confidence>0.4 :
                    class_score = row[5:].max() #maximum probability from 20 objects
                    class_id = row[5:].argmax() #get the index position at which maximum probability occurs

                    if class_score>0.25 : #class_score should be greater than threshold then only enter the if sttaement
                        cx, cy, w, h = row[0:4] # fetch the bounding box for this conditional probability of classs score
                        # construct bounding box from four values
                        # left,  top, width, and height
                        left = int((cx-0.5*w) * x_factor)
                        top = int((cy-0.5*h) * y_factor)
                        width = int(w*x_factor)
                        height = int(h*y_factor)

                        box = np.array([left,top,width,height])

                        #append values into the list
                        classes.append(class_id)
                        confidences.append(confidence)
                        boxes.append(box)

            #clean
            boxes_np = np.array(boxes).tolist()
            confidences_np = np.array(confidences).tolist()

            #NMS
            index = cv2.dnn.NMSBoxes(bboxes=boxes_np, scores=confidences_np, score_threshold=0.25, nms_threshold=0.45).flatten()

            # Step4
            # Draw the bounding box.
            for ind in index:
                # extract bounding box
                x,y,w,h = boxes_np[ind]
                bb_conf = int(confidences_np[ind]*100)
                classes_id = classes[ind]
                class_name = self.labels[classes_id]
                colors = self.generate_colors(classes_id)

                text = f"{class_name}:{bb_conf}%"
                cv2.rectangle(image,(x,y),(x+w,y+h),colors,2)
                cv2.rectangle(image,(x,y-30),(x+w,y),colors,-1) 
                #(255,255,255)-white color, -1 to fill all the rectangular boxes


                cv2.putText(image, text,(x,y-10), cv2.FONT_HERSHEY_PLAIN, 0.7,(0,0,0),1) #(0,0,0)- black color
                
            return image
        
        def generate_colors(self,ID):
            np.random.seed(10)
            colors = np.random.randint(100,255,size=(self.nc,3)).tolist()
            return tuple(colors[ID])
        
    except Exception as e:
        raise ObjectDetectionException(e,sys)
            
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
