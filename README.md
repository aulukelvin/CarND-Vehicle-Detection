## CarNd Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[imagefixa]: ./examples/fit[0].png
[imagefixb]: ./examples/fit[1].png
[imagefixc]: ./examples/fit[2].png
[image1]: ./examples/car_not_car.png
[image1]: ./examples/car_not_car.png
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
### Introduction
#### File Structure
|  Folder and File                     | Description |
|-----|-----|
|/CarND-Vehicle-Detection-P5           | root folder | 
|    /camera_cal/                      |  camera caliberation images, copied from CarND-Advanced-Lane-Finder project|
|    /examples/                        |  screenshots and images for documenting|
|    /test_images/                     |  test images for evaluating detection performance|
|    /utility/                         |  python modules|
|        /p4.py                        |  python module for lane line finding|
|        /p5.py                        |  python module for vehicle detection|
|    /CarND_Advanced_lane_finder.ipynb |  note book for testing advanced lane finder|
|    /VehicleDetector-P5.ipynb         |  note book for vehicle detection|
|    /camera_cals.p                    |  pickled camera caliberation metrics|
|    /perspective_transformation.p     |  pickled perspective transformation metrics|
|    /project_video.mp4                |  original target video to process|
|    /project-out-processed.mp4        |  output video processed using manual feature extraction plus SVM|
|    /project-out-yolo.mp4             |  output video processed using YOLO|
|    /README.md                        |  the document|

### Enhancement from the Advanced Lane Finder
This project was based on the CarND Advanced Lane Finder project. Before started work on the Vehicle Detection project I have done several enhancement on the Advanced Lane Finder scripts. The Advanced Lane Finder project script and some foundamental data have been copied over from the CarND-Advanced-Lane-finder project:

* Filter out exceptional lane line fix result. The produced line fix function is like the following:
```
   f(y) = a*y^2 + b*y + c.    --we use axis y to calculate distance.
```
  I found out it's very effective to evaluate the performance of line fixing. Just for this project video as an example, the parameter a should be between 0.0004 and -0.0005, and the parameter b should be between 0.1 and -0.5. Any other value outside the range can cause the lane line bent too much. When I detected the out range fit result I simple discard the current result and replace it with the last normal figure. The treated result can be depicted as the following:
  
 |  param a | param b  | param c |
 |-----|-----|-----|
 | ![fit param a][imagefixa]  |     ![fit param b][imagefixb] |  ![fit param c][imagefixc]|    

* Level off the impact of extreme values. I used sliding window of 5 latest poly fit results to predict the real parameter. The result shows that the quality of the lane fiting has been improved alot.

* I also re-tuned the perspective transormation. I found out even lift the upper border of the transformation mapping corners a small number of piXels can still loop in much more lane line information. So it will be much easier for the algorithm to fit the lane line. 

The code of Lane line finding are in CarND_Advanced_lane_finder.ipynb and also utility/p4.py  

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

