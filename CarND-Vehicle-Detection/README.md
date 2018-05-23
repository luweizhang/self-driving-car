### Vehicle Detection Project

#### The goal of this project is create a data pipeline to detect vehicles on the road for a self driving car.

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
[myimage4]: ./output_images/model_performance.png
[myimage5]: ./output_images/bounding_boxes.png


### 1. Histogram of Oriented Gradients (HOG)

We use a method called histogram of oriented gradients to extract features from these images.  HOG is a computer vision technique that works by counting the occurrence of gradient orientation in localized portions of an image:  

Before implementing the HOG feature extraction, we first read in the dataset.  Below is a visualization of some of the samples from the dataset:

![alt text][myimage1]

Using the OpenCV library, we implement a method called `extract_hog_features` which takes as input images and HOG parameters and outputs a flattened HOG feature vector for each image in the dataset.  Before extract the HOG features, I tried converting the image to various color spaces including RGB, HSV, LUV, Lab, HLS and YUV.  Through various iterations, it seems like YUV performed the best so we ended up using that in the model.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

![alt text][myimage2]

After generating the flattened HOG features, we normalize the features with zero mean and unit variance using scikit-learns `StandardScaler()` Below is a visualization of the features before and after this normalization process:

![alt text][myimage3]

### 2. Training the classifier

Next, these feature vectors are combined with a label vector (1 for cars, 0 for non-cars) to be used for training the model.  The data is shuffled and split into training and test sets.   

A number of models from sci-kit library are trained and tested to determine the optimal classifier to use on the pipeline. The models that I tried out included logistic regression, support vector machines, and a neural network (multilayered preceptron)

![alt text][myimage4]

### 3. Sliding Window Search

A sliding window approach is taken to classifier the car in the images.   Windows of various sizes will scan the image as the classifier looks for cars.  Instead of performing the HOG feature extraction technique on each of the cars, which would be too computationally expensive, the HOG features are extracted for the entire image, then the subset of these features are fed into the classifier depending on the sliding window.

The overlap in the X direction and Y dierction was set to 75% while the overlap in the Y direction was set to 75%.  This pattern proved fairly effective in producing redundant true positive detections, which is useful for later on weeding out false positive detections using a heatmap strategy (explained below).

![alt text][myimage5]
---

### 4. Video Implementation and Overlapping Thresholds

Finally, use all the steps above to process a video feed for a self driving car.

A deque data structure is used to store bounding boxes from the last 12 frames of the video as it is being processed.  The final list of bounding rectangles will be generated from the last 12 frames of the video instead of just using one frame.   

The OpenCV function cv2.groupRectangles is used to group overlapping boxes together.  A threshold of 10 is used, meaning a minimum of 10 overlapping rectangles must occur before a detection is made.   Doing this serves to weed out false positives and make the model more robust, as it unlikely for more than 10 out of the last 12 frames to contain false positives detections.

Here's a [link to my video result](https://www.youtube.com/watch?v=VuNE1vu05aU)

---

### Discussion

The most difficult parts of this project included:

- Figuring out an effective method for eliminating false positives.  I had to find the appropriate threshold for the model as well as tune the sliding window behavior (i.e window size, overlap percentage) to optimize for the least amount of false positives.  To further improve the model, I could build a much larger dataset by downloading more images or augmenting the dataset.

- Figuring out a way to create discrete bounding boxes on each car.  I had to do alot of experimentation until I found the optimum threshold to apply to the heatmap and optimum strategy for grouping together the boxes.

- Implementation of the pipeline in realtime.  Although the video was only 46 seconds, it took my laptop 31 minutes to process the video feed and detect the vehicles.   Therefore, my model would need a 40x speed up in order to run on realtime on my laptop.   Its not clear to me yet whether increased computational power could enable realtime deployment of this model




