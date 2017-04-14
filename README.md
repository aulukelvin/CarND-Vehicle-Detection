## CarNd Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[imagefixa]: ./examples/fit[0].png
[imagefixb]: ./examples/fit[1].png
[imagefixc]: ./examples/fit[2].png
[imagefixa_old]: ./examples/fit[0]_old.png
[imagefixb_old]: ./examples/fit[1]_old.png
[imagefixc_old]: ./examples/fit[2]_old.png
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
### File Structure
|  Folder and File                     | Description |
|-----|-----|
|/CarND-Vehicle-Detection-P5           | root folder | 
|    /camera_cal/                      |  camera calibration images, copied from CarND-Advanced-Lane-Finder project|
|    /examples/                        |  screenshots and images for documenting|
|    /test_images/                     |  test images for evaluating detection performance|
|    /utility/                         |  python modules|
|    /utility/p4.py                        |  python module for lane line finding|
|    /utility/p5.py                        |  python module for vehicle detection|
|    /CarND_Advanced_lane_finder.ipynb |  note book for testing advanced lane finder|
|    /VehicleDetector-P5.ipynb         |  note book for vehicle detection|
|    /camera_cals.p                    |  pickled camera calibration metrics|
|    /perspective_transformation.p     |  pickled perspective transformation metrics|
|    /project_video.mp4                |  original target video to process|
|    /project-out-processed.mp4        |  output video processed using manual feature extraction plus SVM|
|    /project-out-yolo.mp4             |  output video processed using YOLO|
|    /README.md                        |  the document|

### Enhancement from the Advanced Lane Finder
This project is based on the CarND Advanced Lane Finder project. Before started work on the Vehicle Detection project I have done several enhancement on the Advanced Lane Finder scripts. The Advanced Lane Finder projectede script and some fundamental data have been copied over from the CarND-Advanced-Lane-finder project:

* Filter out exceptional lane line fix result. The produced line fix function is like the following:
   ```
   f(y) = a*y^2 + b*y + c.    --we use axis y to calculate distance.
   ```
  I found out it's very effective to evaluate line fixing performance. Just for this project video as an example, the parameter a should be between 0.0004 and -0.0005, and the parameter b should be between 0.8 and -0.5. Any other value outside the range can cause the lane line bent too much. When I detected the out range fit result I simple discard the current result and replace it with the last normal figure. These threshold may need to be generalized rather than overfit to this particular project but I believe this is the right way to control the robustness of the line fiting. 
  
  The treated result can be depicted as the following:
 
  Fit parameters before trimming exception values:
  
  |  param a | param b  | param c |
  |-----|-----|-----|
  | ![fit param a_old][imagefixa_old]  |     ![fit param b_old][imagefixb_old] |  ![fit param c_old][imagefixc_old]|    


  Fit parameters after trimming exception values:
  
  |  param a | param b  | param c |
  |-----|-----|-----|
  | ![fit param a][imagefixa]  |     ![fit param b][imagefixb] |  ![fit param c][imagefixc]|    
 

* Level off the impact of extreme values by using the average of the poly fit parameters of 5 last frames to make the fitted lines smoother. 

* I also fine tuned the perspective transformation. I found out even if just lift a small number of pixels of the upper border of the transformation mapping window can still loop in much more lane line information, and it will make it much easier for the algorithm to fit the lane line. This is easy to understand, the road information become condensed when it is closer to the vanishing point so even 10 - 20 pixels in the far end can cover quite long distance. 
   The old and new src/dst settings of the perspective transformation are as the following:
   ```python
    #old parameters
    src = np.float32([[565, 470],[720, 470],[290,680],[1090,680]])
    dst = np.float32([[290, 100],[1090, 100],[290,680],[1090,680]])
    
    #new parameters
    src = np.float32([[602, 460],[715, 460],[290,690],[1080,690]]) 
    dst = np.float32([[350, 100],[900, 100],[350,680],[900,680]])
   ```

The code of Lane line finding are in CarND_Advanced_lane_finder.ipynb and also utility/p4.py  

### Vehicle detection pipeline
This project is more for practising computer vision skills so I'm not going to use MLP in the project. Image is very high dimentional data which is very difficult for non-MLP model to analysis. In order to make it easier, normally we need to manually extract features from the image and then seed them into a model like SVM to do classification. Then we use sliding window technique to scan through the whole image to identify target vehicles. Then we'll need to find a way to reduce the noise of the detection. 

I'll explain each step as following:

### Feature extraction
In this project I 
### Training SVM

### Vehicle detection

### False positive reduction

### Video processing

### Discussion
1. Speed vs accuracy trade off
   In the process of tuning, I noticed that linear SVM is much faster than RBF kernel and the sub-sampling HOG extraction is also much faster but the RBF + non-sub-sampling sliding window can produce way better result. Increase the size of HOG orientations can also enhance the performance but the speed will also downgrade accordingly. In this project I choose to bias to accuracy rather than speed and the final selection is RBF + hog orientation=13 + no-sub-sampling. It takes 3.6s - 4s to process a image on my Macbook.

2. How to utilize hardware 
   The code using OpenCV and SKLearn as the corner stones. But both of them can't automatically utilize multi-core CPU and also can't use GPU as well. This can be a big problem if we like to deploy the model to a real system. One of the bottle neck of the performance is the large number of sliding windows. It has 384 windows in each frame under the current settings. I believe using thread pool to make the sliding window analysis runs on all eight cores rather than a single one can be a big step forward. Limited by time, it hasn't finished yet. The HOG feature extraction may benefit from GPU, so may be replace sklearn with some other HOG library which support GPU can be an option.

2. Manual feature extraction plus SVM vs MLP
   This project is more focused on computer vision technologies but it's still good to know what if we simply leave the task to a MLP. I found out leading objection detection solutions, such as YOLO, can directly detect things from the image and the performance is super fast. It can process 67 frames per second using YOLOv2 and can process 207 frames per second using TinyYOLO.
   Besides the speed, we can see that the YOLO result is very stable and able to catch very small objects which is really challenging for SVM method. 
   If we are going to develop a real product then MLP may be the right way to go.

### Histogram of Oriented Gradients (HOG)

The code for HOG feature extraction is contained in the utility/p5.py, line 213 - 235.  

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

