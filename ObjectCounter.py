
'''
Object Counter class.
'''

# pylint: disable=missing-class-docstring,missing-function-docstring,invalid-name

import multiprocessing
import json
import cv2
import numpy as np
from joblib import Parallel, delayed

from tracker import add_new_blobs, remove_duplicates, update_blob_tracker

from Counter import attempt_count


NUM_CORES = multiprocessing.cpu_count()
classes = None




def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    '''
    Draws on top of current frame the bounding bo an specifies label of the object.
    '''

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_output_layers(net):
    '''
    Gets all the output layer of yolo model.
    '''
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def get_bb_yolo(image,net,conf_threshold=0.5,scale=0.00392,nms_threshold=0.4):
    '''
    Returns the bounding box predictions, classes and the corresponding confidence levels.
    '''

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    (H, W) = image.shape[:2]
    #(H,W)=1080,1920
    outs = net.forward(get_output_layers(net))
    
    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes=[]
    confidences=[]
    classIDs=[]

    for output in outs:

        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID in [1,2,3,4,5,6,7,8]:
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > conf_threshold:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
        nms_threshold)

    final_bounding_boxes = []
    final_classes = []
    final_confidences = []

    if len(idxs) > 0:
            # print("Detected BBs : ",len(idxs))
            # loop over the indexes we are keeping
        for i in idxs.flatten():

            # extract the bounding box coordinates
            final_bounding_boxes.append(boxes[i])
            final_classes.append(classIDs[i])
            final_confidences.append(confidences[i])
            
            


    return final_bounding_boxes,final_classes,final_confidences

class ObjectCounter():

    def __init__(self, frame, net, mcdf, mctf, di, counting_line,class_file,tracker='kcf',conf_threshold=0.5\
        ,nms_threshold=0.4,scale=0.00392):
        self.frame = frame # current frame of video
        self.net = net
        self.mcdf = mcdf # maximum consecutive detection failures
        self.mctf = mctf # maximum consecutive tracking failures
        self.detection_interval = di
        self.counting_line = counting_line
        
        with open(class_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        self.blobs = {}
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.counts = {counting_line['label']: {}} # counts of objects by type for each counting line
        self.tracker = tracker
        self.conf_threshold=conf_threshold
        self.nms_threshold=nms_threshold
        self.scale=scale
        # self.
        
        
        # create blobs from initial frame
        _bounding_boxes, _classes, _confidences = get_bb_yolo(frame, net,conf_threshold,scale,nms_threshold)
        self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)

    def get_counts(self):
        return self.counts

    def get_blobs(self):
        return self.blobs

    def count(self, frame):
        self.frame = frame

        blobs_list = list(self.blobs.items())
        # update blob trackers
        blobs_list = Parallel(n_jobs=NUM_CORES, prefer='threads')(
            delayed(update_blob_tracker)(blob, blob_id, self.frame) for blob_id, blob in blobs_list
        )
        self.blobs = dict(blobs_list)

        for blob_id, blob in blobs_list:
            # count object if it has crossed a counting line
            blob, self.counts = attempt_count(blob, blob_id, self.counting_line, self.counts,self.classes)

            self.blobs[blob_id] = blob

            # remove blob if it has reached the limit for tracking failures
            if blob.num_consecutive_tracking_failures >= self.mctf:
                del self.blobs[blob_id]

        if self.frame_count >= self.detection_interval:
            # rerun detection
            
            _bounding_boxes, _classes, _confidences = get_bb_yolo(self.frame,self.net,self.conf_threshold\
                ,self.scale,self.nms_threshold)

            self.blobs = add_new_blobs(_bounding_boxes, _classes, _confidences, self.blobs, self.frame, self.tracker, self.mcdf)
            self.blobs = remove_duplicates(self.blobs)
            self.frame_count = 0

        self.frame_count += 1

    def visualize(self):
        frame = self.frame
        (H, W) = frame.shape[:2]
        #(H,W)=1080,1920
        # draw and label blob bounding boxes
        for _id, blob in self.blobs.items():

            (x, y, w, h) = [int(v) for v in blob.bounding_box]
            x_plus_w = x+w
            y_plus_h = y+h

            draw_prediction(frame, blob.type, blob.type_confidence, x, y, x_plus_w, y_plus_h,self.classes,self.COLORS)

        # draw counting line
        cv2.line(frame, self.counting_line['line'][0], self.counting_line['line'][1],(255,255,255), 3)
        
        temp_i=0
        for line in self.counts:
            cv2.putText(frame,line, (int(0.90*W),int(0.8611*H)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            for vehicle,count in self.counts[line].items():
                temp_i=temp_i+30
                cv2.putText(frame,"{}={}".format(vehicle,count), (int(0.90*W),int(0.8611*H+(temp_i/1080)*H))\
                    ,cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
                
                

        #json.dumps(self.counts)
        return frame

    