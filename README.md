### Udacity self driving car projects:

### Guide to setting up tensorflow
http://luweilikesdata.blogspot.com/2018/01/installing-tensorflow-for-cpu-and-gpu.html

### CarND-Term1-Starter-Kit-Test
```
# Anaconda
source activate carnd-term1 # If currently deactivated, i.e. start of a new terminal session
jupyter notebook test.ipynb
```

```
# Docker
docker run -it --rm -p 8888:8888 -v ${pwd}:/src udacity/carnd-term1-starter-kit test.ipynb
# OR
docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit test.ipynb
```

### CarND-LaneLines-P1 (Car Street Lane Detection)
Build a data pipeline to automatically detect lanes from a video feed.  Use canny edge detection to find edges in the image (using convolutional filters to find the gradients).  Apply a region mask to isolate the lanes, then apply Hough Transformation to calculate the final lines which represent the position of the lanes.  

Hough Transformation is a feature extraction technique, typically used for detecting lines from images.  It works by apply a point-to-curve transformation (A hough transformation), and then using a polling technique to find the best fit line for the given points. As mentioned before, Hough Transformation works by converting a line to a point.  The point is representated as y = mx+b, where m and b are the x and y coordinates.  Noticed that in Hough Space, all points along the same line will have the same m, although they might have different b.  In addition, a point in cartesian space will be a line in Hough Space.  This is becuase a point in cartesian space will have infinitely many slopes in Hough Space, so it is represented as a line.  Notice that when you have many points in cartesian space, the points that fall along the same line will be represented as a bunch of lines intersecting in Hough Space.  This intersection is where you will find the best fit line for your street lanes!  (I know this explanation is a bit complicated without diagrams and such, I guess I will add those later)

### CarND-Traffic-Sign-Classifier-Project
Build a deep learning pipeline to classify German street signs.  Use LeNet convolutional neural net architecture.  

Utilize data augmentation to deal with the small and imbalanced dataset.  

### CarND-Behavioral-Clone
Train a convolutional neural network using Keras to drive a car in a simulator by mimicking human behavior.  Data will be collected by driving the car in the simulator.   The final result is that the car will be able to predict the optimum steering angle given an image. 

### Additional Useful Commands

list available virtual environments  
```conda env list```
