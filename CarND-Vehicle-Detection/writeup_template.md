### Vehicle Detection Project

The goal of this project is create a data pipeline to detect vehicles on the road for a self driving car.

The steps for creating this data pipeline are roughly as follows:

1.  Perform a histogram of oriented gradients (HOG) feature 
extraction process on a labeled training set of images.

2.  Use the output of the HOG to train a supervised classifier (SVM, logistic regression, neural network, etc.) 

3.  Implement a sliding window technique with windows of various 
sizes using the trained classifier to search for vehicles in the images using the classifier.

4.  Create a overlap threshold to reject false positives.  Also estimate a bounding box based on pixels detected.


[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[myimage1]: ./output_images/dataset.png
[myimage2]: ./output_images/hog_visualization.png
[myimage3]: ./output_images/features.png

---

### Histogram of Oriented Gradients (HOG)

We use a method called histogram of oriented gradients to extract features from these images.  HOG is a computer vision technique that works by counting the occurrence of gradient orientation in localized portions of an image:  

Before implementing the HOG feature extraction, we first read in the dataset.  Below is a visualization of some of the samples from the dataset:

![alt text][myimage1]

We implement a function called `extract_features` which does this.  Before extract the HOG features, I tried converting the image to various color spaces including RGB, HSV, LUV, Lab, HLS and YUV.  Through various iterations, it seems like YUV performed the best so we ended up using that in the model.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][myimage2]

After generating the flattened HOG features, we normalize the features with zero mean and unit variance using scikit-learns `StandardScaler()` Below is a visualization of the features before and after this normalization process:

![alt text][myimage3]
\

### 2. Training the classifier

A number of models from sci-kit library are trained and tested to determine the optimal classifier to use on the pipeline. The models that I tried out included logistic regression, support vector machines, and a neural network (multilayered preceptron)



### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

For the sliding window search, I experimented with various sizes and 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

The most difficult parts of this project included:

- Figuring out an effective method for eliminating false positives.  I had to find the appropriate threshold for the model as well as tune the sliding window behavior (i.e window size, overlap percentage) to optimize for the least amount of false positives.  To further improve the model, I could build a much larger dataset by downloading more images or augmenting the dataset.

- Figuring out a way to create discrete bounding boxes on each car.  I had to do alot of experimentation until I found the optimum threshold to apply to the heatmap and optimum strategy for grouping together the boxes.

- Implementation of the pipeline in realtime.  Although the video was only 46 seconds, it took my laptop 31 minutes to process the video feed and detect the vehicles.   Therefore, my model would need a 40x speed up in order to run on realtime on my laptop.   Its not clear to me yet whether increased computational power could enable realtime deployment of this model




