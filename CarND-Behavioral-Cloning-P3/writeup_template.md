I am teaching a car how to drive in a 3d simulator by cloning human behavior.

In this project, a convolutional neural network (CNN) is trained from the raw pixels of a car's camera feed, and eventually learns to associate this data with the correct steering angle in the training data.  The training data is gathered from human driving, with both the camera feed and steering angle being recorded during data collection.

The end to end deep learning approach differs from traditional approaches. Traditional approaches generally involve an explicit decomposition of the problem, such as lane marking detection, path planning, and control.  In contrast, the end to end deep learning approach involves automatically learning how to drive by detecting useful road features from raw video feed data and training against the steering angle of the car.

A paper published by NVidia describes this approach in more detail:
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf


----------------


# Data Collection:
I will generate my training data by driving this car manually through the simulated track and collecting data from the 3 cameras at the front of the car as well as recording the steering angle of the car.  I will then preprocess and augment this data before training a convolutional neural network.


The distribution of the raw data collected (x axis is steering angle and y axis is number of observations).   As you can see, most of the data comes from instances when the car is in a neutral or near neutral position.  In order for the model to train more effectively, we might need to balance this dataset with data augmentation


Looking at the distribution above, you can see that the dataset is heavily skewed towards low or zero angles.  This is because during data collection, most of the simulated roads were straight roads.  This could be a problem during training if neural network does not have enough data to learn how to handle turns correctly.  To handle this problem, I threw away observations whose steer angle was under a certain threshold with a probability of p.


# Data Preprocessing:

After collecting the data, I performed some data preprocessing.  Because the angle coming from each camera was slightly different, I added an offset of either .25 or -.25 to the steering angle for the left and right camera respectively.  In addition, images produced by the simulator had a resolution of 320x160, but the CNN expect input images to be 200x66.  To achieve this size, I cropped the bottom 20 pixels and top 35 pixels and then resized to 200x66.      During prediction time, the same preprocessing is applied to new images before being fed into the CNN for prediction.

(more coming soon)


# Data Augmentation:
I augmented the dataset by introducing "jitter."  Jitter consists of a randomized brightness adjustment, randomized shadow, and horizon shift.  The horizon shift is used to mimick uphill or downhill situations where the horizon from the perspective of the car would change.

In addition, images were flipped horizontally to simulate driving in the other direction.

Data augmentation increases the data size and helps the model generalize better when it comes time to test.
(more coming soon).

# Model Architecture:
The architecture of the CNN will be adapted from the NVidia paper linked above. First, I will recreate the CNN architecture developed by the NVidia team using Keras, a Python library for deep learning.  The CNN architecture consists of  three 5x5 convolution layers, followed by two 3x3 convolution layers, followed by three fully connected layers.  A RELU activation function is used in each of the fully connected layers.  During training, an Adam optimizer was used and the loss function was mean squared error (MSE).  In addition, a dropout of 20% was used.
(more coming soon)




The architecture of the CNN used by the NVidia team to train 

The architecture consists of three 5x5 convolution layers, followed by two 3x3 convolution layers, followed by three fully connected layers.


Below is how the model architecture was defined in Keras.  Keras code is extremely concise and allows you to quickly define complex model architectures. 
```
 def create_model(dropout=.2, activation=ELU()):
    """Define the CNN architecture"""
    model = Sequential()  

    # Normalize  
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    
    #crop
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    
    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride  
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    #model.add(Dropout(dropout))
    # Add two 3x3 convolution layers (output depth 64, and 64)  
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))  
    model.add(ELU())  
    # Add a flatten layer  
    model.add(Flatten())  
    # Add three fully connected layers (depth 100, 50, 10 neurons respectively), elu activation
    model.add(Dense(100, W_regularizer=l2(0.001)))  
    model.add(ELU())  
    #model.add(Dropout(dropout))
    model.add(Dense(50, W_regularizer=l2(0.001)))  
    model.add(ELU())  
    #model.add(Dropout(dropout))
    model.add(Dense(10, W_regularizer=l2(0.001)))  
    model.add(ELU())  
    #model.add(Dropout(dropout))
    # Add a fully connected output layer  
    model.add(Dense(1))  
    # Compile and train the model,   
    model.compile(optimizer=Adam(lr=1e-4), loss='mse') 
    
    return model
```
Notice that ELU activation functions are used, as it tends to be better at avoiding the vanishing gradient problem:
see here for justification: http://saikatbasak.in/sigmoid-vs-relu-vs-elu/


### Handling the large dataset using generators.
Python generators are especially useful when processing big data.  Using generators allows you to load the dataset one batch at a time rather than loading it all at once.

Here is an example of how a generator can be used in Keras:

```
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(ch, row, col),
        output_shape=(ch, row, col)))
model.add(... finish defining the rest of your model architecture here ...)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= /
            len(train_samples), validation_data=validation_generator, /
            nb_val_samples=len(validation_samples), nb_epoch=3)
```
# Results.
Finally I tested the performance of my car which will make steering decision based on the output of the CNN.
The means squared loss (MSE) of the model on the validation set decreasing with each epoch during training.  After about 3 epochs, the validation loss stopped improving at around .08

(more coming soon)

# Additional Resources:

CNN architecture used by the comma.ai team: https://github.com/commaai/research/blob/master/train_steering_model.py

A paper published by NVidia describing the end-to-end deep learning approach for self driving cars:
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
