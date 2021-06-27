# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:46:30 2021

@author: Sonu
"""

import numpy as np
import cv2
        
def stitch(images, ratio=0.75, reprojThresh=4.0, showMatches=False):
    #Unpacking the images to get query and train image
    (imageB, imageA) = images
    #Detecting keypoints and extract local invariant descriptors from the images
    (kpsA, featuresA) = detectAndDescribe(imageA)
    print(np.shape(kpsA))
    print(np.shape(featuresA))
    (kpsB, featuresB) = detectAndDescribe(imageB)
    print(np.shape(kpsB))
    print(np.shape(featuresB))
    
    #Matching features between the two images
    M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)
    #if the match is None, then there aren't enough matched keypoints to create a panorama
    if M is None:
        return None
    
    (matches, H, status) = M
    
    #Warping the query image to align it along the same plane as the train image
    #Bringing both the images in the same perspective view
    #result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    #Better Image Rendering
    result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1] - int(imageB.shape[1]/2), imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
    
    #Checking to see if the keypoint matches should be visualized
    if showMatches:
        vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)
    
    return result
    
def detectAndDescribe(image):
    #Creating a Descriptor object
    descriptor =  cv2.xfeatures2d.SIFT_create()
    #Extracting the keypoints and descriptors with SIFT
    (kps, features) = descriptor.detectAndCompute(image, None)
    #Converting the keypoints from KeyPoint objects to NumPy arrays
    kps = np.float32([kp.pt for kp in kps])
    
    return (kps, features)
    
def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    #Matching object using Brute force matcher
    #Brute force matcher takes feature from query image and exhaustively comapres it with all the features in the train image. 
    #Uses the eucildean distance to perform the feature match
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    
    #Finding the top two matches for each descriptor using KNN match
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        #Store all the good matches as per Lowe's ratio test.
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    print(len(matches))
    #Computing a homography requires at least 4 matches
    if len(matches) > 4:
        #Finding the keypoints for good matches only
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        #Finding the homography between keypoints of matching pair sets using RANSAC
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        
        return (matches, H, status)
    
    #If at least 4 matches are not found return none
    return None
    
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    #Initializing the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB
    
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        #Only process the match if the keypoint was successfully matched
        if s == 1:
            #Drawing the match
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
    
    return vis
        