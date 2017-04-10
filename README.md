# Vehicle Detection

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image2]: ./examples/non.png
[image3]: ./examples/car_HOG.png
[image4]: ./examples/non_HOG.png
[image5]: ./examples/heatmap.png
[image6]: ./examples/heatmap_clean.png
[image7]: ./examples/labels.png
[image8]: ./examples/boxes.png
[image9]: ./examples/project_video.gif

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in in `get_HOG_features()` at line 179 of `utils.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

<img src="https://github.com/dyelax/CarND-Vehicle-Detection/blob/master/examples/car.png?raw=true" width=49% />
<img src="https://github.com/dyelax/CarND-Vehicle-Detection/blob/master/examples/non.png?raw=true" width=49% />

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=12`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


<img src="https://github.com/dyelax/CarND-Vehicle-Detection/blob/master/examples/car_HOG.png?raw=true" width=49% />
<img src="https://github.com/dyelax/CarND-Vehicle-Detection/blob/master/examples/non_HOG.png?raw=true" width=49% />

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters while training my classification model, and empirically found that the parameters mentioned aboce resulted in the highest accuracy.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG, color bin and color histogram features in `train_model()` at line 668 of `utils.py`. To save on iteration time, I saved the trained models so I could load them without retraining. My final model was a Linear SVM trained on `HLS` images, the aforementioned HOG settings, color bin features of shape `(16, 16)` and color histogram features with 32 bins. This resulted in a test accuracy of 90.3%. I trained on the GTI data and tested on the other directories to avoid the risk of overfitting that came with having similar images in the same directory.

Some other settings resulted in higher test accuracy. For example, training on images in `YCrCb` space gave a test accuracy of 93.2%; however, upon qualitative inspection of performance one the test video, these settings did not perform as well. I suspect this is because the color spaces perform differently on the training images and the frames of the test video, which have obvious qualitative differences.  

### Sliding Window Search

#### Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented sliding window search in `get_sliding_window_preds` at line 368 in `utils.py`. I searched on windows of scales `64x64` and `96x96` over the whole road area (y pixels 400-656), because these searches are relatively quick and produce good results. I also searched at scale `32x32`, limited to y pixels 400-500, because this search took much longer, and smaller car detections are more likely to be closer to the horizon. Here is an example of a heatmap of detections made on a test image:

![Heatmap][image5]

#### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I filtered false positives by implementing a rolling threshold on the heatmap. To do this, I zeroed out any pixels that had fewer than 4 heatmap detections over the past 8 frames. This resulted in the following cleaned heatmap.

![Heatmap clean][image6]

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.

<img src="https://github.com/dyelax/CarND-Vehicle-Detection/blob/master/examples/labels.png?raw=true" width=100% />

From there, I constructed bounding boxes to cover the area of each blob detected.

![Boxes][image8]

---

### Video Implementation

#### Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

<img src="https://github.com/dyelax/CarND-Vehicle-Detection/blob/master/examples/project_video.gif?raw=true" width=100% />

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Initially, I ran into a lot of issues with overfitting. My model was getting 99.8% test accuracy, but was giving terrible bounding box predictions. This was because. Even after fixing this by splitting train/test data by directory, some of my models that performed well in test accuracy resulted in poor bounding box predictions on the test video. I believe this is because the train/test data and the test video are significantly different in terms of image quality / saturation. Another issue is that the sliding window detection is currently too slow to be run in real time. In the future I would like to speed this up and improve performance by using a CNN classifier on a GPU. I would also like to combine this with the advanced lane finding project to create a full lane and car detection pipeline.

