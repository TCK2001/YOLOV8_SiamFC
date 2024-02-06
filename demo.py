from __future__ import absolute_import

import os
import glob
import numpy as np
import argparse
import cv2

from siamfc import TrackerSiamFC

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='modelfolder for demo',
                    required=True)

parser.add_argument('--data', help='Choose Video folder in data directory data\\Car1',
                    required=True)

args = parser.parse_args()
modeldir = args.model
datadir = args.data


CWD_PATH = os.getcwd()

if datadir:
    PATH_TO_DATA = os.path.join(CWD_PATH,datadir)
    print(PATH_TO_DATA)
    videodata = glob.glob(PATH_TO_DATA)[0] 
    print(videodata)

if modeldir:
    PATH_TO_DATA = os.path.join(CWD_PATH,modeldir)
    modelpth = glob.glob(PATH_TO_DATA + '/*')[0] #pretrained
    print(modelpth)

video_path = './video/bird.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    exit()

ret, img = cap.read()
cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

# setting ROI
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
print(rect)
cv2.destroyWindow('Select Window')

#initialize tracker
#tracker = TrackerSiamFC(net_path=modelpth)
#tracker.init(img,rect)
count = 0
name_num=0
while True:
    ret, frame = cap.read()
    if not ret :
        exit()

    count += 1
    if count <10 :
        name_num ='000'+str(count)
    if count >9 and count <100:
        name_num = '00'+str(count)
    if count >99 and count <1000:
        name_num = '0'+str(count)
    if count >999:
        name_num=str(count)
    image_path = './video/img/' + str(name_num) + '.jpg'
    cv2.imwrite(image_path, frame)
    print(image_path)

    seq_dir = os.path.expanduser(videodata)
    img_files = sorted(glob.glob(seq_dir + '/img/*.jpg'))
    tracker = TrackerSiamFC(net_path=modelpth)
    tracker.track(img_files, rect, visualize=True)
    if cv2.waitKey(1) == ord('q'):
        break
