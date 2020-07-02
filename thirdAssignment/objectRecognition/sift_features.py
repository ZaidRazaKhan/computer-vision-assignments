import cv2
import numpy as np

img = cv2.imread('image_2.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# sift = cv2.SIFT()
kp = sift.detect(gray,None)
kp, des = sift.detectAndCompute(gray,None)
img=cv2.drawKeypoints(gray ,kp ,img)
# kp, des = sift.detectAndCompute(gray,None)
cv2.imwrite('sift_keypoints.jpg',img) 