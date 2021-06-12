
import cv2
import argparse
import numpy as np
import imutils
import time
import os
from ObjectCounter import ObjectCounter
from tracker import add_new_blobs, remove_duplicates, update_blob_tracker,_kcf_create,get_tracker

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--video', required=True,
                help = 'path to input video')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-o', '--output_video', required=True,
                help = 'output video file name')
args = ap.parse_args()

scale = 0.00392        # Scaling for preprocessing frame before feeding to yolo
DI=10                  # Detection intervals
conf_threshold = 0.5   # Threshold for selecting label of vehicle
nms_threshold = 0.4    # Threshold for Non Max Suppression
mcdf=2                 # Max Continous Detection Failures
mctf=2                 # Max Continous Tracking Failures
percentage=45          # Resizing parameter
counting_line={'label': 'Bottom', 'line': [(0,880), (1920,880)]} # The line used for counting as vehicles pass it 
tracker='kcf'                                                    # The type of tracker used between DI
class_file=args.classes
    
net = cv2.dnn.readNet(args.weights, args.config)                 # The CV2 network used for detection
 
vs = cv2.VideoCapture(args.video)

writer = None
(W, H) = (None, None)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# Try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
  


# Read first frame
success, frame = vs.read()

(H, W) = frame.shape[:2]           #(H,W)=1080,1920

counting_line={'label': 'Bottom', 'line': [(0,int(0.815*H)), (W,int(0.815*H))]}

# Object Counter object is created and tracks all blobs untill they move out of screen
object_counter = ObjectCounter(frame, net, mcdf, mctf,
                                   DI, counting_line,class_file,tracker)


frames_processed = 0               # The number of frames out of 'total' that have been processed

while success:

    start= time.time()
    # timer = cv2.getTickCount()
    object_counter.count(frame)                 # Count the number of objects that have crossed the counting line 
    output_frame = object_counter.visualize()   # Returns a frame with the counting line and bounding box drawn on it

        
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args.output_video,cv2.CAP_OPENCV_MJPEG, fourcc, 30,
        (int(frame.shape[1]*percentage/100), int(frame.shape[0]*percentage/100)), True)
        # some information on processing single frame
        end= time.time()
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(
                elap * total))
    # write the output frame to disk
    if output_frame is not None:
        output_frame=rescale_frame(output_frame,percentage)
        writer.write(output_frame)

    success, frame = vs.read()


# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()
print(object_counter.counts)
        
