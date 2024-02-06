from __future__ import absolute_import

import argparse
import os
import glob
import cv2

from siamfc import TrackerSiamFC

parser = argparse.ArgumentParser()
parser.add_argument('--data', help='Video file folder path', required=True)
parser.add_argument('--model', help='Model file folder path', required=True)

args = parser.parse_args()
datadir = args.data
modeldir = args.model
CWD_PATH = os.getcwd()

if datadir:
    PATH_TO_DATA = os.path.join(CWD_PATH,datadir)
    videodata = glob.glob(PATH_TO_DATA)[0] 
    print(videodata)

if modeldir:
    PATH_TO_DATA = os.path.join(CWD_PATH,modeldir)
    modelpth = glob.glob(PATH_TO_DATA + '/*')[0] 
    print(modelpth)

# 홈 디렉토리에 저장.    
seq_dir = os.path.expanduser(videodata)

img_files = sorted(glob.glob(seq_dir + '/*.jpg'))

# anno = np.loadtxt(seq_dir + '/groundtruth_rect.txt',delimiter=',')

anno=[]
img = cv2.imread(img_files[0])

cv2.namedWindow('Let\'s Tracking')
cv2.imshow('Let\'s Tracking', img)

# setting ROI
rect = cv2.selectROI('Let\'s Tracking', img, fromCenter=False, showCrosshair=True)
anno.append(rect)
print(anno[0])

cv2.destroyWindow('Let\'s Tracking')

if __name__ == '__main__':
    print('start')
    seq_dir = os.path.expanduser(videodata)
    img_files = sorted(glob.glob(seq_dir + '/*.jpg'))
    #anno = np.loadtxt(seq_dir + '/groundtruth_rect.txt',delimiter=',')
    net_path = modelpth
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)


"""
from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = os.path.expanduser('~/data/OTB/Crossing/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    
    net_path = 'pretrained/siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)
"""