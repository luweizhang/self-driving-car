## Summary
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"
[img1]: ./output_images/undistort.png "Undistorted"

All the correspomding code for this writeup can be found in pipeline.ipynb

### Writeup / README

### Camera Calibration

The first step is to calibrate the camera using images of chess boards.  
We have 20 images of chessboards with 9x6=54 internal corners that we can use to calibrate the camera
(It is recommended that we use at least 20 images to calibrate the camera.) 

First, we create a 3d array called objp which represents the 3 dimensional coordinate
location of all the the chessboard corners. We then initialize `objpoints` to store the aforementioned 3d points as well as `imgpoints` to store the 2d chesspoint corners that we find.

We use opencv to convert the images to grayscale, we then use `cv2.findChessboardCorners()`
to obtain the chessboard corners from the image. We then append these results to objpoints and imgpoints

In the next step, we create an `undistort()` function that uses the objpoints and imgpoints that we obtained
from the previous step to calibrate the camera.   After the camera has been calibrated, we can use the parameters 
obtained to undistort all the road images using cv2.undistort(img, mtx, dist, None, mtx)

Below, I have visualized the original (distorted) and undistorted images.  The difference between the distorted and 
undistorted image is not immediately obvious, as most of the distortion appears on the edges of the image.  However, if you look closely at the edges of the image, you can see the areas where the distortion has been corrected

![alt text][img1]

### Pipeline (single images)

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Perspective Transform

The code for the perspective transform in contained in the function `perspective_transform`.  The source and destination points were hard coded into the function mostly through eyeballing and trial and error.  The parameters can be seen below:

```
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    
    #destination points were calculated visually.
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
 ```
 
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's are links to my vidoe result:
- [github](./result.mp4)
- [youtube](https://www.youtube.com/watch?v=Es8ZrvnxjYs)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
