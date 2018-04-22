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
[img2]: ./output_images/color-threshold.png "Color Thresholding"
[img3]: ./output_images/perspective-transform.png "Perspective Transform"
[img4]: ./output_images/fill-lane.png "Perspective Transform"


All the correspomding code for this writeup can be found in pipeline.ipynb

### Writeup / README


### Description of Pipeline:

#### 1. Camera Calibration

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


#### 2. Perspective Transform

The code for the perspective transform in contained in the function `perspective_transform`.  The source and destination points were hard coded into the function mostly through eyeballing and trial and error.  The parameters can be seen below:

```
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    
    #destination points were calculated visually.
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
 ```
 
![alt text][img3]

#### 3. Color Thresholding

After applying the perspective transform, we use color thresholding to isolate the pixels from the lane lines.

For the color thresholds, I isolated the "lightness" channel from the hls colorspace
and the "blue-yellow" channel from the lab colorspace

The "lightness" from the hls channel was very useful for isolating the white lane lines and the "blue-yellow"
channel from the lab colorspace was very good for getting the yellow lines.

I had to test out a number of thresholds before I got the optimal ones.  I tested out different channels from 
various color spaces including LUV, HSV, HLS, LAB, and I tried various gradient thresholds.  Ultimately, 
I did not use the gradient thresholds and simply used channels from two colorspaces.

After that, I combined the two binary maps to get the final isolated lane lines.

You can see from the results below that the thresholds chosen do a pretty good job of isolating the lane lines 
![alt text][img2]


#### 4. Detecting the lane line pixels and fitting a line to the pixels

The next step is to detect the lane lines.

In the function `fit_lane_lines()` , lane lines are detected by identifying 
peaks in a histogram on the bottom portion of the image and detecting nonzero pixels in close proximity to the peaks.

After the pixels have been identified, a sliding windows approach is used to detect the pixels for the entire line.  
Once all the pixels have been identified for both the left and right lanes, a best fit line is calculated using 
a second order polynomial using `np.polyfit()`.

The vehicle center is calculated by average the x coordidinates of the left and right lane lines
and then finding the midpoint between the two

After the lane lines have been fit and drawn onto the image, a reverse perspective transform is applied to get the 
final result.

![alt text][img4]

#### 5. Calculate radius of curvature

The radius of curvature is also calculated in the `fit_lane_lines()` function described above.

The formula for radius of curvature can be found here: https://www.intmath.com/applications-differentiation/8-radius-curvature.php

The formula is:
```
curve_radius = ((1 + (2*fit[0]*y_0*y_meters_per_pixel + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

#### 6. Apply a reverse perspective transform to get the final result

After applying all the steps above, a reverse perspective transform is applied to achieve the final result, which looks something like below:

![alt text][image6]

---

### Pipeline (video)

#### 1. Final video output:

Here's are links to my vidoe result:
- [github](./result.mp4)
- [youtube](https://www.youtube.com/watch?v=Es8ZrvnxjYs)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The hardest part of the project was definitely getting familiar with the different tools, such as numpy and opencv to get the desired result.  I had to experiment with countless parameters and read alot of reference materials before I could get a good grasp of how to implement the pipeline. 

One issue I encountered was preventing the fitted line from "wobbling" too much across subsequent frames in the video.  I resovled this by using a moving average of the lane line pixels across the last n frames.  I implemented a class called `Line` which stores the parameter for both the left and right lane lines.


