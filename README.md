

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

Hough Transformation is a feature extraction technique, typically used for detecting lines from images.  It works by apply a point-to-curve transformation (A hough transformation), and then using a polling technique to find the best fit line for the given points. 

### CarND-Traffic-Sign-Classifier-Project
Build a deep learning pipeline to classify German street signs.  Use LeNet convolutional neural net architecture.

### Additional Useful Commands

list available virtual environments  
```conda env list```
