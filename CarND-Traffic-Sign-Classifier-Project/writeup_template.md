# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

(see jupyter notebook for examples)

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/luweizhang/self-driving-car/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

(please see ipynb, I used seaborn to create a histogram of the class frequency)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale.  To convert to grayscale, I took the average of the three color channels.  I believe that the main benefit of converting to grayscale is that it simplifies the input and reduces the computational requirements without much loss of performance (Although I think this claim is still an active area of research and not for certain).

(see ipython notebook for example pictures)

Next, I normalized the image data because it standardizes the input which I believe allows for faster training during backwards propagation.  (I am still not exactly sure what the benefit it / if there is any benefit at all of normalizing your dataset before training.)

Finally, the dataset is class imbalanced.  Some of class have less than 300 examples while some have 2000.  One way of addressing this is to use data augmentation.  To augment the data you can apply different types of transformations to the image, such as rotation, zooming in and out, clipping, translation, and lighting adjustments.  I have not augmented my data yet so the results below are trained on an unaugmented dataset.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model is a convolutional neural network based off of the LeNet architecture.  The Lenet architecture is a way of architecting a convolutional neural network which consists of a series of convlution layers, activation layers, and pooling layers, and finally a fully connected layer:

Here is an example of what a RELU architecture might look like.
INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC => RELU => FC

Here are some details of my model architecture:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale and Normalized image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, input = 32x32x1, output = 28x28x48 	|
| Max pooling	      	| 2x2 stride,  Input = 28x28x48. Output = 14x14x48				|
| Convolution 5x5	    | 1x1 stride, Output = 10x10x96    									|
| Max pooling	      	| 2x2 stride,  Input = 10x10x96. Output = 5x5x96				|
| Convolution 3x3	    | 1x1 stride, Output = 3x3x172									|
| Max pooling	      	| 2x2 stride,  Input = 3x3x172. Output = 2x2x172			|
| Flatten     	| Input = 2x2x172. Output = 688.		|
| Fully connected		| Input = 688. Output = 84         	|
| Fully connected		| Input = 94. Output = 43 									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tested a number of different optimizers, including stochastic gradient descent, rmsprop, and adam optimizer.  I also tried out a number of batch sizes from 64 to 256, and tried out epoch lengths ranging from 25 to 100.  For the learning rate, I chose .001. 

From my limited experimentation, the hyperparameters that seem to result in the best performance is
learning_rate = .001
epoch = 50 (don't see much more improvement with more training)
batch_size = 128
optimizer = adam or rmsprop (gradient descent trains too slowly)

Some things that I have yet to try:
- dropout
- inception layers
- different mu and sigma parameters

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


If an iterative approach was chosen:
* The first architecture that I tried was a typical lenet architecture taken from the udacity examples.  
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

My final model results were:
* training set accuracy of 
* validation set accuracy of 97.1
* test set accuracy of 95.1

I started out by trying out the well known LENET architecture.  I chose lenet because it is easy to implement and it has been to be effective in image classification problems.  From there I pretty much used trial and error and my intution.  I started experimenting with different hyperparameters such as the learning rate, number of epochs, batch size, and the optimzer.  I kept a log model performance with different combinations of hyperparameters and iterated towards the best possible result.

Todo for the future:
- try out different well known architectures  Googlnet?  Alexnet? Transferlearning

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
(please see my ipython notebook)

Some of the images might be hard to classify due to a number of reasons:
- sign partially clipped off
- bad lighting
- sign is at an angle

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
(please see ipython notebook for further details) 

Aside from the "unknown" traffic sign, my model was able to predict all the examples correctly.  The traffic signs that had bad lighting, at an angle, or partially clipped off were classified correctly. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

(Please see ipython notebook for more details).

The softmax probabilities for my model were pretty much unambigious.  Aside from the "priority road" sign which was partially clipped, the softmax probability was close to 100% for winning prediction.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

...haven't done this yet...

