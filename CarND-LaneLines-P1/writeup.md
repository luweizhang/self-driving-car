# **Finding Lane Lines on the Road** 
by Luwei Zhang
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. 

1. Convert the image to grayscale to make lanes easier to detect
2. Add a Gaussian blur to remove any minor artifacts
3. Add a region mask to isolate the location of the lanes
4. Apply Hough Transformation to identify the best fit line segments from the lane pixels. 
Then pool these results to find the best overall line to fit the left and right lane.
5. Merge the line back to the original image.

https://www.youtube.com/watch?v=VtoOmQZPKUM

In order to draw a single line on the left and right lanes, The draw_lines() function was modified to use the cv2.poly_fit() function.  Instead of doing a raw Hough Transformation, the best fit line segments were classifed into two groups, either the left or right lane, based on their slope.   The left lane has a positive slope from the POV of the camera, the right lane is vice versa (negative slope).  

After these two groups have been identified, then a best fit line is found using the cv2.poly_fit() function.  This best fit line is used to draw the final two lines identifying the lanes

<put some images here>

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


A potential short coming of my pipeline is that it doesn't handle edge cases very well.  I'm not sure how it will perform if say a piece of trash is seen on the road, or if a car is directly in front.  

Another shortcoming is if the car is in an offroad situation, or on a road if unclear lane markings.

Another short coming is that my pipeline relies on calculating two best fit lines.  However, if there are three lanes it might get confused.  It would be interesting to see how the pipeline would perform under these conditions.


### 3. Suggest possible improvements to your pipeline

One possible improvement to the pipeline is adding some sort of memory feature, so that if the camera is obstructed, it will make guesses at where the lanes are.  However, this is probabably something to be calculated further downstream in the pipeline and might not be necessary in this process.



