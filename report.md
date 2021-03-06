**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_YCrCb.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/pipeline.jpg
[image5]: ./output_images/detection.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./output_images/HOG_RGB.jpg
[video1]: ./project_video_result.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #44 through #152 of the file called `feature.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:
![alt text][image8]

In RGB color space, we see almost no changes in HOG among the channels. However, the changes are more evident in `YCrCb` color space:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and mainly tried to detect cars properly in the video pipeline. The final parameters are the following:

|Parameter|Value|
|:--------|----:|
|Color Space|YCrCb|
|HOG Orient|9|
|HOG Pixels per cell|8|
|HOG Cell per block|2|
|HOG Channels|All|

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM (`sklearn.svm.LinearSVC`) using both HOG features and color features (Line #1 to #45 at `training.py`). All the features are normalized (function defined at line 29 to 37 in `training.py` and called from the notebook). This is very important critical step. Otherwise, classifier may have some bias toward to the features with higher weights.

Randomly selected 20% of images are used for testing (lines 23-24 in `training.py`).

For a particular run, SVM training log looks like:
```
Feature vector length: 8460
7.72 Seconds to train SVC...
Test Accuracy of SVC =  0.9885
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search in right half portion of the image in different scales `[ 1.0, 1.5, 2.0 ]` using Hog Sub-sampling Window Search technique. The overlap was 75%. The decision was made on a trial and error basis based on the observed performance in detection accuracy. (Code resides at lines 8 to 95 in `detection.py`)


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which gave more than 98% accuracy on randomly sampled test data. So no further optimization was tried. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected
(code is available at `detection.py`).


### Here are six frames and their corresponding heatmaps and resulting bounding boxes:
![alt text][image5]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main problem I faced is eliminating the false positives. I struggled for days and was partially successful.

Also to avoid false positives, image scan starts only from (500, 400) upto right bottom corner of the image. So the detection will not generalize.

Better algorithm like YOLO (You Only Look Once) should be robust to improve this situation.



