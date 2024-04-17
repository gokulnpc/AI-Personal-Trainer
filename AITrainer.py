import cv2
import numpy as np
import os
import PoseEstimationModule as pm
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

pTime = 0
cTime = 0
detector = pm.poseDetector()

count = 0
direction = 0

while True:
    lmList = []
    success, img = cap.read()
    # img = cv2.imread('train.jpg')
    img = cv2.resize(img, (1280, 720))
    img = cv2.flip(img, 1)
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    if lmList:
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 300), (0, 100))
        bar = np.interp(angle, (210, 300), (650, 100))
        # Check for the dumbbell curls 
        # Up
        if per == 100:
            if direction == 0:
                count += 0.5
                direction = 1
        # Down
        if per == 0:
            if direction == 1:
                count += 0.5
                direction = 0
                
        cv2.rectangle(img, (0, 450), (200, 720), (0, 255, 0), cv2.FILLED)
        cv2.rectangle(img, (0, 450), (200, int(bar)), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (40, 600), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        cv2.putText(img, f'{int(count)}', (40, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)
        
        
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    