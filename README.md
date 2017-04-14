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
[image2]: ./examples/HOG_example.png
[image3]: ./examples/car_detected.png
[image4]: ./examples/car_missed.png
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
In this project I extracted three kinds of features for the SVM to use: the color histogram, the spatial histogram, and the HOG(Histogram of Oriented Gradient). Both the color histogram and the spatial histogram are extracting color features from the image.

The spatial histogram can be considered as an enhanced version of color template matching. It generalize the image into small scalled template so that it can produce more common features than too detailed full image. If can works better than pure color template matching but the drawback is still the same: it cannot scales from training color to other unfamiliar color, and it will be heavily impacted by lumination, etc. Compare to other features, spatial histogram is more sensitive to the position of the object.  

The code for spatial histogram extraction is in the /utility/p5.py as the following:

   ```python
      def bin_spatial(img, size=(32, 32)):
         # Use cv2.resize().ravel() to create the feature vector
         features = cv2.resize(img, size).ravel() 
         # Return the feature vector
         return features
   ```
The color histogram is just simply calculate the histogram of the whole image for each color channel and concate the three histograms into a single vector. Again, it's depends on the color but because it extracting features of color pattern rather than the real layout of the image, it is more stable than spatial histogram. But the downside is it also lacks enough detail to solely support making decision.

The code for color histogram is in the /utility/p5.py as the following:
   ```python
      def color_hist(img, nbins=32, bins_range=(0, 256)):
         # Compute the histogram of the color channels separately
         channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
         channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
         channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
         # Concatenate the histograms into a single feature vector
         hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
         # Return the individual histograms, bin_centers and feature vector
         return hist_features

   ```
   
Then the most important feature is the HOG, which takes a gray scaled image or a single color channel from the color image and calculate the histogram of gradient orientations for small pixel patches. The Hog feature is more highlighting the profile of the  image. Then the SVM can use the HOG to identify the structure of the vehicle.

The HOG example is in the /utility/p5.py like the following:

![HOT_example][image2]

The code is as below:

      ```python
         from skimage.feature import hog
         
         features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
         return features, hog_image
      ```

It shows that HOG feature itself can contribute most of the features but it is still benefial to include color histogram and color histogram to produce even better result.

### Training SVM
The next step is training the SVM. Udacity already provided a labeled training set which has 8792 car images and 8968 non-car images. All images are cropped to 64x64 pixels. The example images are as the following:

![Car_nocar][image1]

Then I'll need to extract features from all the car and non-car images to produce the dataset. Then I do a train_test_split to preserve 20% of the data as test set and only train the model using the 80% training set. After the training finished then use the test set to evaluate the performance of the model.

There're heaps of parameter and super-parameters to tune in this step. The first important one is the color space. We can choose from RGB, HSV, LUV, HLS, YUV, YCrCb. And we can also choose to use all color channels or only one of them. Then we can choose whether to include spatial histogram, color histogram, or HOG. Then we can choose subtle parameters to make the model performes better. After long time of trail and error, I choose the following settings:

```python
   color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
   orient = 13  # HOG orientations
   pix_per_cell = 16 # HOG pixels per cell
   cell_per_block = 2 # HOG cells per block
   hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
   spatial_size = (16, 16) # Spatial binning dimensions
   hist_bins = 16    # Number of histogram bins
   spatial_feat = True # Spatial features on or off
   hist_feat = True # Histogram features on or off
   hog_feat = True # HOG features on or off
```
I found out the performance can increase along with the orient settings but the training and testing speed will be downgraded. Increasing pix_per_cell can speed up the speed and also leads to better performance. I chose orient=13 as the final because of the better performance and the speed is also acceptable. 

Last but not the least, I need to select best SVM kernel as well. I tried Linear kernel and RBF kernel, and even tried grid search best combination of the kernel parameters. I noticed that Linear kernel can performance twice as fast as the RBF, but the test accuracy is lower than RBF. When compare the models with real image later, the gap is even bigger. I choose the RBF because of it performance. But the speed is really not good. When processing real image, the average speed is 3.8s per frame. 

The final performance of the test is:
65.63 Seconds to train SVC...
Test Accuracy of SVC =  0.9975
15.3 Seconds to predict.
 
The code can be found from VehicleDetector-P5.ipynb section four and five.

### Vehicle detection
All the training images are 64X64. The way we do vehicle detection on real image is using a technology called sliding window to scan through the image to see which part is matching a vehicle. This idea is quite simple but it has a couple problems. The first one is more related to the template matching itself. The quality of the prediction is heavily depends on how similar the testing scenario is to the testing data. If the positioin of the car is not so popular in the training set, or the color is rare then the result may not be accurate. From the test images we can see the detection of the black car is much easier than detection of the white car. I believe this can be improved by put in more different colored training data and do augmentation.

The second problem is because the vehicle appear in the image in all scales so we have to scan the image use different scales to capture both large vehicle patches and also small ones. But that can leads to one more problem: the number of the sliding windows can be large so that the speed will be very slow. 

To limit the number of sliding windows, I have limited the searching area to between the vanishing point and the car bonet. I also tried different sizes of the sliding windows and found out using both 64X64 and 128X128 can produce good enough result without downgrade the speed too much. The final number of sliding windows for a single frame is 384. This is a actually a tradeoff between performance and speed. I didn't use smaller window size that means much less windows but also means the model will not able to catch small objects.

A more advanced technique to improve the detection speed is called HOG sub-sampling. Nomal HOG computation has lot of redundant computation becaue there are lot of overlap between celles. The HOG sub-sampling calculates the HOG for whole image in a single shot and then produce feature by takes sub-area values. This method proved to be able to double the speed but the accuracy is also degraded. So I didn't include HOG sub-sampling in the final model because of the time limit.  

### False positive reduction
The SVM is very powerful to catch objects but it can often raise false alarm as well. For example the model will often been confused by the lane line and predicts lot of inexist vehicles along the line. One trick to fix this is called hard negtive mining, which means identify the failed detections, add them into the training data and retrain the model. Another trick is called heatmap, which means let the overlapping windows to vote if that area contains an object. If the result is over some threshold then we consider it's possitive.  

The result is like the following:
Car detected:
![Car detected][image3]

Car missed:
![Car missed][image4]

The code for this part can be found from /CarND-Vehicle-Detection.ipnb section 8. 
### Video processing
The Video processing is basically reusing the Advanced Lane Finder project. I included the binary threshold, bird-eye view, and heatmap in the processed video for evaluation. 

Here's a [link to my video result](./project_video_processe.mp4)

### Discussion
1. Speed vs accuracy trade off
   In the process of tuning, I noticed that linear SVM is much faster than RBF kernel and the sub-sampling HOG extraction is also much faster but the RBF + non-sub-sampling sliding window can produce way better result. Increase the size of HOG orientations can also enhance the performance but the speed will also downgrade accordingly. In this project I choose to bias to accuracy rather than speed and the final selection is RBF + hog orientation=13 + no-sub-sampling. It takes 3.6s - 4s to process a image on my Macbook.

2. How to utilize hardware 
   The code using OpenCV and SKLearn as the corner stones. But both of them can't automatically utilize multi-core CPU and also can't use GPU as well. This can be a big problem if we like to deploy the model to a real system. One of the bottle neck of the performance is the large number of sliding windows. It has 384 windows in each frame under the current settings. I believe using thread pool to make the sliding window analysis runs on all eight cores rather than a single one can be a big step forward. Limited by time, it hasn't finished yet. The HOG feature extraction may benefit from GPU, so may be replace sklearn with some other HOG library which support GPU can be an option.

2. Manual feature extraction plus SVM vs MLP
   This project is more focused on computer vision technologies but it's still good to know what if we simply leave the task to a MLP. I found out leading objection detection solutions, such as YOLO, can directly detect things from the image and the performance is super fast. It can process 67 frames per second using YOLOv2 and can process 207 frames per second using TinyYOLO.
   Besides the speed, we can see that the YOLO result is very stable and able to catch very small objects which is really challenging for SVM method. 
   If we are going to develop a real product then MLP may be the right way to go.

