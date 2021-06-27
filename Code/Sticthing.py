# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:12:47 2021

@author: Sonu
"""
from Panaroma import stitch
import cv2

# load the two images and resize them to have a width of 400 pixels for faster processing
imageA = cv2.imread('Image6_A.png')
imageB = cv2.imread('Image6_B.png')
imageA_Height = imageA.shape[0]
imageA = cv2.resize(imageA, (400, imageA_Height))
imageB = cv2.resize(imageB, (400, imageA_Height))

# stitch the images together to create a panorama
(result, vis) = stitch([imageA, imageB], showMatches=True)

# show the images
cv2.imshow("Image A", imageA)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imshow("Image B", imageB)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imshow("Keypoint Matches", vis)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imshow("Result", result)
cv2.waitKey(20000)
cv2.destroyAllWindows()
