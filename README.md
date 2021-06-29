# Panoramic image stitching
This Repo gives a demonstration of implementation of stitching of overlapping images using Homography Matrix. With the change in the width of the output image during perspective warping, the rendering of the Panoramic images was more refined.

### Input images
![Left image](Input%20Images/Image1_A.png?raw=true)
![Right image](Input%20Images/Image1_B.png?raw=true)

### Design Pipeline
The Design Pipeline is as follows:
* Read the pair of raw images.
* Initiate a keypoint descriptor object. Here SIFT is used.
* Find the keypoints and descriptors with SIFT.
* Match object using Brute force matcher
* Retrieve the top two matches for each descriptor using KNN match with N=2.
* Store all the good matches as per Lowe's ratio test.
* Detect the keypoints for good matches only.
* Compute the homography between keypoints of matching pair sets using RANSAC.

### Keypoint Matching
![Matched Keypoint](Keypoint%20Matches/Image_1_Keypoint_Matches.png?raw=true)

* Warp the query image to align that image along the same plane as the train image.
* Stitch the two images.

### Stitched Image
![Stitched Image](Better%20Stitch/Image1_Stitch1.png?raw=true)
