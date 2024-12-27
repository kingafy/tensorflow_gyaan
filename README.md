# tensorflow_gyaan
https://github.com/louishenrifranc/Tensorflow-Cheatsheet

https://github.com/SHANK885/Tensorflow-in-Practice-Specialization
https://github.com/ashishpatel26/Tensorflow-in-practise-Specialization/blob/master/4.%20Sequences%20and%20Prediction/w4-quiz.txt
https://github.com/MedAzizTousli/Tensorflow-in-Practice-Specialization/blob/master/Assignments/Course%201%20Week%202.ipynb

https://github.com/NishantBhavsar/ML-with-Tensorflow-and-GCP

https://www.datacamp.com/community/tutorials/ten-important-updates-tensorflow

Research.google..com/seedbank---google projects

Colab.research.google.com

GPUusage :-https://jhui.github.io/2017/03/07/TensorFlow-GPU/

TensorFlow is available at TensorFlow.org, and video updates from the TensorFlow team are at youtube.com/tensorflow

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/6tZWF/week-1-resources> 

Play with a neural network right in the browser at http://playground.tensorflow.org. See if you can figure out the parameters to get the neural network to pattern match to the desired groups. The spiral is particularly challenging!
The 'Hello World' notebook that we used in this course is available on GitHub here.

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/6tZWF/week-1-resources> 

https://github.com/ashishpatel26/Tensorflow-in-practise-Specialization/blob/master/3.%20Natural%20Language%20Processing%20in%20TensorFlow/w2-quiz.txt


Really like the focus on practical application and demonstrating the latest capability of TensorFlow. As mentioned in the course, it is a great compliment to Andrew Ng's Deep Learning Specialization

From <https://www.coursera.org/programs/applied-intelligence-specialization-mmddyy-80-0cokg?collectionId=&productId=5ghJ5U8zEemp3woY6REV3A&productType=s12n&showMiniModal=true> 

A step by step explanation of how to use TensorFlow 2.0 for building a Neural network for sequences and time series. With detailed examples of code and of how to choose hyper-parameters.

From <https://www.coursera.org/programs/applied-intelligence-specialization-mmddyy-80-0cokg?collectionId=&productId=5ghJ5U8zEemp3woY6REV3A&productType=s12n&showMiniModal=true> 






One layer 
One neuron
Input shape is onevalue






Training loop will go with 500 times




=====================SIMPLE MODEL=====
You can download the workbook here if you want to try it out for yourself. Or, if you prefer, you can execute it right now in Google Colaboratory at this link.

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/QAMJd/try-it-for-yourself> 


Simple regression problem with TF
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(
  optimizer='sgd',
  loss='mean_squared_error'
)
xs = np.array([1, 2, 3, 4])
ys = np.array([100, 150, 200, 250]) / 100
model.fit(xs, ys, epochs=1000)

print(model.predict([7.0]))



Load datasets available in keras


https://developers.google.com/machine-learning/fairness-overview/





The last layer has 10 neurons as it is a 10 class classification problem

Fashin mnist code:--



Introducing callback to control training---very important to stop training if loss is under our thresholds



Implementation of callbacks :-



import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
callbacks = myCallback()
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])



##MNIST using simple DNN

import tensorflow as tf

class CustomCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
      if(logs.get('acc')>0.99):
        print("\n 99% acc reached")
        self.model.stop_training = True
        
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(units=256, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=128, activation=tf.nn.relu),
        tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history =model.fit(
        x_train,
        y_train,
        epochs=10,
        callbacks=[CustomCallbacks()]
    )


    
    # YOUR CODE SHOULD END HERE

    # model fitting
    return history.epoch, history.history['acc'][-1]
    
    
    # model fitting
    #return history.epoch, history.history['acc'][-1]
Here are all the notebook files for this week, hosted on GitHub. You can download and play with them from there!
Beyond Hello, World - A Computer Vision Example
Exploring Callbacks
Exercise 2 - Handwriting Recognition - Answer

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/0x7b1/week-2-resources> 



CNN
The concepts introduced in this video are available as Conv2D layers and MaxPooling2D layers in TensorFlow. You’ll learn how to implement them in code in the next video…

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/JWQud/coding-convolutions-and-pooling-layers> 

https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D

You’ve seen how to add a convolutional 2d layer to the top of your neural network in the previous video. If you want to see more detail on how they worked, check out the playlist at https://bit.ly/2UGa7uH.
Now let’s take a look at adding the pooling, and finishing off the convolutions so you can try them out…

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/mSVJQ/learn-more-about-convolutions> 




Here’s the notebook that Laurence was using in that screencast. To make it work quicker, go to the ‘Runtime’ menu, and select ‘Change runtime type’. Then select GPU as the hardware accelerator!
Work through it, and try some of the exercises at the bottom! It's really worth spending a bit of time on these because, as before, they'll really help you by seeing the impact of small changes to various parameters in the code. You should spend at least 1 hour on this today!

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/YbOJA/try-it-for-yourself> 



To try this notebook for yourself, and play with some convolutions, here’s the notebook. Let us know if you come up with any interesting filters of your own!
As before, spend a little time playing with this notebook. Try different filters, and research different filter types. There's some fun information about them here: https://lodev.org/cgtutor/filtering.html

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/mLx0E/experiment-with-filters-and-pools> 



Gpu config:--
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


###MNIST using CNN
# GRADED FUNCTION: train_mnist_conv
class myCallback(tf.keras.callbacks.Callback):
    def on_epoc_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.998):
            print("Reached accuracy 99.8% accuracyc so cancelling training!")
            self.model.stop_training = True
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    
    
    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images/255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images/255.0
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images, training_labels, epochs = 10, callbacks = [myCallback()])
    # model fitting
    return history.epoch, history.history['acc'][-1]


We've put the notebooks that you used this week into GitHub so you can download and play with them.
Adding Convolutions to Fashion MNIST
Exploring how Convolutions and Pooling work

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/uu6IH/week-3-resources> 


###complex images
As Andrew and Laurence discussed, the techniques you’ve learned already can apply to complex images, and you can start solving real scenarios with them. They discussed how it could be used, for example, in disease detection with the Cassava plant, and you can see a video demonstrating that here. Once you’ve watched that, move onto the next lesson!

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/O1veW/explore-an-impactful-real-world-solution> 



Image generator helps to label images put in folder




Train_dir is the parent dict

Images shud be of same size
Target size param converts to standard size
Classmode can be binary or depending on more classes





Model.summary() would be



Now that you’ve designed the neural network to classify Horses or Humans, the next step is to train it from data that’s on the file system, which can be read by generators. To do this, you don’t use model.fit as earlier, but a new method call: model.fit_generator. In the next video you’ll see the details of this.

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/vi5Ch/train-the-convnet-with-imagegenerator> 



Now that you’ve learned how to download and process the horses and humans dataset, you’re ready to train. When you defined the model, you saw that you were using a new loss function called ‘Binary Crossentropy’, and a new optimizer called RMSProp. If you want to learn more about the type of binary classification we are doing here, check out this great video from Andrew!

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/zvk4R/training-the-neural-network> 

Google colab :--- details

Now it’s your turn. You can find the notebook here. Work through it and get a feel for how the ImageGenerator is pulling the images from the file system and feeding them into the neural network for training. Have some fun with the visualization code at the bottom!
In earlier notebooks you tweaked parameters like epochs, or the number of hidden layers and neurons in them. Give that a try for yourself, and see what the impact is. Spend some time on this.
Once you’re done, move to the next video, where you can validate your training against a lot of images!

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/rVqWi/experiment-with-the-horse-or-human-classifier> 



Automatic validation 

Now you can give it a try for yourself. Here’s the notebook the Laurence went through in the video. Have a play with it to see how it trains, and test some images yourself! Once you’re done, move onto the next video where you’ll compact your data to see the impact on training.

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/QgonH/get-hands-on-and-use-validation> 

Measuring the dimensions of training data is very crucial

Try this version of the notebook where Laurence compacted the images. You can see that training times will improve, but that some classifications might be wrong! Experiment with different sizes -- you don’t have to use 150x150 for example!

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/SjYWI/get-hands-on-with-compacted-images> 

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab happy-or-sad.zip from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()


# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):         
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('acc')>DESIRED_ACCURACY):
                print("\nReached 99.9% accuracy so cancelling training!")
                self.model.stop_training = True
    

    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
        
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
        

    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_datagen = ImageDataGenerator(rescale=1/255)# Your Code Here

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        "/tmp/h-or-s", 
        target_size=(150, 150), 
        batch_size=10,
        class_mode='binary'
        )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
          train_generator,
      steps_per_epoch=2,  
      epochs=15,
      verbose=1,
      callbacks=[callbacks])
    # model fitting
    return history.history['acc'][-1]

train_happy_sad_model()


You used a few notebooks this week. For your convenience, or offline use, I've shared them on GitHub. The links are below:
Horses or Humans Convnet
Horses or Humans with Validation
Horses or Humans with Compacting of Images

From <https://www.coursera.org/learn/introduction-tensorflow/supplement/sXKJM/week-4-resources> 

In the next video, you'll look at the famous Kaggle Dogs v Cats dataset: https://www.kaggle.com/c/dogs-vs-cats
This was originally a challenge in building a classifier aimed at the world's best Machine Learning and AI Practitioners, but the technology has advanced so quickly, you'll see how you can do it in just a few minutes with some simple Convolutional Neural Network programming.
It's also a nice exercise in looking at a larger dataset, downloading and preparing it for training, as well as handling some preprocessing of data. Even data like this which has been carefully curated for you can have errors -- as you'll notice with some corrupt images!
Also, you may notice some warnings about missing or corrupt EXIF data as the images are being loaded into the model for training. Don't worry about this -- it won't impact your model! :)

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/4ijuU/the-cats-vs-dogs-dataset> 

Cats vs Dogs

Now that we’ve discussed what it’s like to extend to real-world data using the Cats v Dogs dataset from an old Kaggle Data Science challenge, let’s go into a notebook that shows how to do the challenge for yourself! In the next video, you’ll see a screencast of that notebook in action. You’ll then be able to try it for yourself.

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/iNiIM/looking-at-the-notebook> 


Cropping of image helps a lot

Augmentation:--->
You'll be looking a lot at Image Augmentation this week.
Image Augmentation is a very simple, but very powerful tool to help you avoid overfitting your data. The concept is very simple though: If you have limited data, then the chances of you having data to match potential future predictions is also limited, and logically, the less data you have, the less chance you have of getting accurate predictions for data that your model hasn't yet seen. To put it simply, if you are training a model to spot cats, and your model has never seen what a cat looks like when lying down, it might not recognize that in future.
Augmentation simply amends your images on-the-fly while training using transforms like rotation. So, it could 'simulate' an image of a cat lying down by rotating a 'standing' cat by 90 degrees. As such you get a cheap way of extending your dataset beyond what you have already.
To learn more about Augmentation, and the available transforms, check out https://github.com/keras-team/keras-preprocessing -- and note that it's referred to as preprocessing for a very powerful reason: that it doesn't require you to edit your raw images, nor does it amend them for you on-disk. It does it in-memory as it's performing the training, allowing you to experiment without impacting your dataset.

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/I8qgl/image-augmentation> 


Ok, now that we've looked at Image Augmentation implementation in Keras, let's dig down into the code.
You can see more about the different APIs at the Keras site here: https://keras.io/preprocessing/image/

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/3DmTp/start-coding> 

Image agmentation options



Horizontal flipping is like mirror image
Fill mode ise used  for missing pixels

Now that you've gotten to learn some of the basics of Augmentation, let's look at it in action in the Cats v Dogs classifier.
First, we'll run Cats v Dogs without Augmentation, and explore how quickly it overfits.
If you want to run the notebook yourself, you can find it here: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%204%20-%20Lesson%202%20-%20Notebook%20(Cats%20v%20Dogs%20Augmentation).ipynb

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/xG3lW/looking-at-the-notebook> 

Having clearly seen the impact that augmentation gives to Cats v Dogs, let’s now go back to the Horses v Humans dataset from Course 1, and take a look to see if the augmentation algorithms will help there! Here’s the notebook if you want to try it for yourself!

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/5LiP6/try-it-for-yourself> 

# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

# In this exercise you will train a CNN on the FULL Cats-v-dogs dataset
# This will require you doing a lot of data preprocessing because
# the dataset isn't split into training and validation for you
# This code block has all the required inputs
import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd
# This code block unzips the full Cats-v-Dogs dataset to /tmp
# which will create a tmp/PetImages directory containing subdirectories
# called 'Cat' and 'Dog' (that's how the original researchers structured it)
path_cats_and_dogs = f"{getcwd()}/../tmp2/cats-and-dogs.zip"
shutil.rmtree('/tmp')

local_zip = path_cats_and_dogs
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
print(len(os.listdir('/tmp/PetImages/Cat/')))
print(len(os.listdir('/tmp/PetImages/Dog/')))

# Expected Output:
# 1500
# 1500
# Use os.mkdir to create your directories
# You will need a directory for cats-v-dogs, and subdirectories for training
# and testing. These in turn will need subdirectories for 'cats' and 'dogs'
to_create = [
    '/tmp/cats-v-dogs',
    '/tmp/cats-v-dogs/training',
    '/tmp/cats-v-dogs/testing',
    '/tmp/cats-v-dogs/training/cats',
    '/tmp/cats-v-dogs/training/dogs',
    '/tmp/cats-v-dogs/testing/cats',
    '/tmp/cats-v-dogs/testing/dogs'
]

for directory in to_create:
    try:
        os.mkdir(directory)
        print(directory, 'created')
    except:
        print(directory, 'failed')
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
# YOUR CODE STARTS HERE

    all_files = []
    
    for file_name in os.listdir(SOURCE):
        file_path = SOURCE + file_name

        if os.path.getsize(file_path):
            all_files.append(file_name)
        else:
            print('{} is zero length, so ignoring'.format(file_name))
    
    n_files = len(all_files)
    split_point = int(n_files * SPLIT_SIZE)
    
    shuffled = random.sample(all_files, n_files)
    
    train_set = shuffled[:split_point]
    test_set = shuffled[split_point:]
    
    for file_name in train_set:
        copyfile(SOURCE + file_name, TRAINING + file_name)
        
    for file_name in test_set:
        copyfile(SOURCE + file_name, TESTING + file_name)
# YOUR CODE ENDS HERE


CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
print(len(os.listdir('/tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('/tmp/cats-v-dogs/testing/dogs/')))

# Expected output:
# 1350
# 1350
# 150
# 150
# DEFINE A KERAS MODEL TO CLASSIFY CATS V DOGS
# USE AT LEAST 3 CONVOLUTION LAYERS
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(150, 150, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
# YOUR CODE HERE
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
TRAINING_DIR = '/tmp/cats-v-dogs/training'
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
)

VALIDATION_DIR = '/tmp/cats-v-dogs/testing'
validation_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=.2,
    height_shift_range=.2,
    shear_range=.2,
    zoom_range=.2,
    horizontal_flip=True,
    fill_mode='nearest'

)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    batch_size=64,
    class_mode='binary',
    target_size=(150, 150)
)


# Expected Output:
# Found 2700 images belonging to 2 classes.
# Found 300 images belonging to 2 classes.
history = model.fit_generator(train_generator,
                              epochs=2,
                              verbose=1,
                              validation_data=validation_generator)
# PLOT LOSS AND ACCURACY
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")


plt.title('Training and validation loss')

# Desired output. Charts with training and validation metrics. No crash :)

Now that we've seen the concepts behind transfer learning, let's dig in and take a look at how to do it for ourselves with TensorFlow and Keras.
In the next few videos you'll be using this notebook to explore transfer learning: https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb
For more on how to freeze/lock layers, explore the documentation, which includes an example using MobileNet architecture: https://www.tensorflow.org/tutorials/images/transfer_learning

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/7TST4/start-coding> 






Transfer learning:-



Another useful tool to explore at this point is the Dropout.
The idea behind Dropouts is that they remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!
Check out Andrew's terrific video explaining dropouts here: https://www.youtube.com/watch?v=ARq74QuavAo

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/GtDVB/using-dropouts> 

Now that we've explored transfer learning, and taken a look at regularization using dropouts, let's step through the scenario for Cats vs Dogs in this notebook next.

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/G6RPR/applying-transfer-learning-to-cats-v-dogs> 

Exercise 3 - Horses vs. humans using Transfer Learning
This week your exercise will be to apply what you've learned about Transfer Learning to see if you can increase training accuracy for Horses v Humans. To avoid crazy overfitting, your validation set accuracy should be around 95% if you do it right!
Your training should automatically stop once it reaches this desired accuracy.
Let's now use Transfer Learning to increase the training accuracy for Horses v Humans!

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/notebook/GpKYz/exercise-3-horses-vs-humans-using-transfer-learning> 

TRANSFER LEARNING CODE:--
# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

# Import all the necessary files!
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd
path_inception = f"{getcwd()}/../tmp2/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

# Import the inception model  
from tensorflow.keras.applications.inception_v3 import InceptionV3

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = path_inception

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None)# Your Code Here

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
      layer.trainable = False# Your Code Here
  
# Print the model summary
pre_trained_model.summary()

# Expected Output is extremely large, but should end with:

#batch_normalization_v1_281 (Bat (None, 3, 3, 192)    576         conv2d_281[0][0]                 
#__________________________________________________________________________________________________
#activation_273 (Activation)     (None, 3, 3, 320)    0           batch_normalization_v1_273[0][0] 
#__________________________________________________________________________________________________
#mixed9_1 (Concatenate)          (None, 3, 3, 768)    0           activation_275[0][0]             
#                                                                 activation_276[0][0]             
#__________________________________________________________________________________________________
#concatenate_5 (Concatenate)     (None, 3, 3, 768)    0           activation_279[0][0]             
#                                                                 activation_280[0][0]             
#__________________________________________________________________________________________________
#activation_281 (Activation)     (None, 3, 3, 192)    0           batch_normalization_v1_281[0][0] 
#__________________________________________________________________________________________________
#mixed10 (Concatenate)           (None, 3, 3, 2048)   0           activation_273[0][0]             
#                                                                 mixed9_1[0][0]                   
#                                                                 concatenate_5[0][0]              
#                                                                 activation_281[0][0]             
#==================================================================================================
#Total params: 21,802,784
#Trainable params: 0
#Non-trainable params: 21,802,784
last_layer = pre_trained_model.get_layer('mixed7')# Your Code Here
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output# Your Code Here

# Expected Output:
# ('last layer output shape: ', (None, 7, 7, 768))

# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97):
      print("\nReached 97.0% accuracy so cancelling training!")
      self.model.stop_training = True

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)# Your Code Here)
# Add a dropout rate of 0.2
x = layers.Dropout(.2)(x)  # Your Code Here                
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x) # Your Code Here)          

model = Model( pre_trained_model.input, x) # Your Code Here

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
                metrics=['accuracy'])

model.summary()

# Expected output will be large. Last few lines should be:

# mixed7 (Concatenate)            (None, 7, 7, 768)    0           activation_248[0][0]             
#                                                                  activation_251[0][0]             
#                                                                  activation_256[0][0]             
#                                                                  activation_257[0][0]             
# __________________________________________________________________________________________________
# flatten_4 (Flatten)             (None, 37632)        0           mixed7[0][0]                     
# __________________________________________________________________________________________________
# dense_8 (Dense)                 (None, 1024)         38536192    flatten_4[0][0]                  
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 1024)         0           dense_8[0][0]                    
# __________________________________________________________________________________________________
# dense_9 (Dense)                 (None, 1)            1025        dropout_4[0][0]                  
# ==================================================================================================
# Total params: 47,512,481
# Trainable params: 38,537,217
# Non-trainable params: 8,975,264


# Get the Horse or Human dataset
path_horse_or_human = f"{getcwd()}/../tmp2/horse-or-human.zip"
# Get the Horse or Human Validation dataset
path_validation_horse_or_human = f"{getcwd()}/../tmp2/validation-horse-or-human.zip"
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import zipfile
import shutil

shutil.rmtree('/tmp')
local_zip = path_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/training')
zip_ref.close()

local_zip = path_validation_horse_or_human
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/validation')
zip_ref.close()


# Define our example directories and files
train_dir = '/tmp/training'
validation_dir = '/tmp/validation'

train_horses_dir = os.path.join(train_dir, 'horses') 
train_humans_dir = os.path.join(train_dir, 'humans') 
validation_horses_dir = os.path.join(validation_dir, 'horses')
validation_humans_dir = os.path.join(validation_dir, 'humans')

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)# Your Code Here

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1/255 )# Your Code Here

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
    batch_size=64,
    class_mode='binary',
    target_size=(150,150))# Your Code Here    

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
    batch_size=64,
    class_mode='binary',
    target_size=(150,150)
        )
# Your Code Here)

# Expected Output:
# Found 1027 images belonging to 2 classes.
# Found 256 images belonging to 2 classes.


# Run this and see how many epochs it should take before the callback
# fires, and stops training at 97% accuracy

##callbacks = 
history = model.fit_generator(
    train_generator,
    epochs=3,
    validation_data=validation_generator,
    callbacks=[myCallback()]
)


%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


Exercise 3 - Horses vs. humans using Transfer Learning
This is the same exercise and notebook as provided here. This button below will take you to the Google Colaboratory environment, in case you would like to use it to follow along with the course videos. In order to pass the graded item, you will still need to submit your work via the Coursera-hosted Jupyter Notebook.
----------------------------------------------------------------------------------
This week your exercise will be to apply what you've learned about Transfer Learning to see if you can increase training accuracy for Horses v Humans to 99.9% or greater. To avoid crazy overfitting, your validation set accuracy should be around 95% if you do it right!
Your training should automatically stop once it reaches this desired accuracy, and it should do it in less than 100 epochs. Running on a colab GPU, I've been able to hit this metric in about 3 minutes and 69 epochs, and I'm sure with a bit of trial and error you could do much better!
For an increased challenge, see if you can get the validation set to 99% or above also! :)
Let' now use Transfer Learning to increase the training accuracy for Horses v Humans!

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/ungradedLti/68LiD/exercise-3-horses-vs-humans-using-transfer-learning> 



##MULTICLASS CLASSIFICATION

http://www.laurencemoroney.com/rock-paper-scissors-dataset/
Rock Paper Scissors is a dataset containing 2,892 images of diverse hands in Rock/Paper/Scissors poses. It is licensed CC By 2.0 and available for all purposes, but it’s intent is primarily for learning and research.
Rock Paper Scissors contains images from a variety of different hands,  from different races, ages and genders, posed into Rock / Paper or Scissors and labelled as such. You can download the training set here, and the test set here. These images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images. I also generated a few images that you can use for predictions. You can find them here.
Note that all of this data is posed against a white background.
Each image is 300×300 pixels in 24-bit color

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/HoOHN/introducing-the-rock-paper-scissors-dataset> 



For multiclass we have to change parameter to categorical









In the following video you'll see me testing the classifier myself. You should also try it out using the data that you can find here.

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/supplement/ytouT/try-testing-the-classifier> 

# ATTENTION: Please do not alter any of the provided code in the exercise. Only add your own code where indicated
# ATTENTION: Please do not add or remove any cells in the exercise. The grader will check specific cells based on the cell position.
# ATTENTION: Please use the provided epoch values when training.

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd

def get_data(filename):
  # You will need to write code that will read the file passed
  # into this function. The first line contains the column headers
  # so you should ignore it
  # Each successive line contians 785 comma separated values between 0 and 255
  # The first value is the label
  # The rest are the pixel values for that picture
  # The function will return 2 np.array types. One with all the labels
  # One with all the images
  #
  # Tips: 
  # If you read a full line (as 'row') then row[0] has the label
  # and row[1:785] has the 784 pixel values
  # Take a look at np.array_split to turn the 784 pixels into 28x28
  # You are reading in strings, but need the values to be floats
  # Check out np.array().astype for a conversion
    with open(filename) as training_file:
      # Your code starts here
        reader = csv.reader(training_file, delimiter=',')    
        imgs = []
        labels = []

        next(reader, None)
        
        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))

            imgs.append(img)
            labels.append(label)

        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
            
    
      # Your code ends here
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../tmp2/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../tmp2/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

# Keep these
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

# Their output should be:
# (27455, 28, 28)
# (27455,)
# (7172, 28, 28)
# (7172,)


# In this section you will have to add another dimension to the data
# So, for example, if your array is (10000, 28, 28)
# You will need to make it (10000, 28, 28, 1)
# Hint: np.expand_dims

training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    # Your Code Here
    )

validation_datagen = ImageDataGenerator(
    rescale=1 / 255)
    # Your Code Here)
    
# Keep These
print(training_images.shape)
print(testing_images.shape)
    
# Their output should be:
# (27455, 28, 28, 1)
# (7172, 28, 28, 1)

# Define the model
# Use no more than 2 Conv2D and 2 MaxPooling2D
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')]
    
    )
    # Your Code Here

# Compile Model. 
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    # Your Code Here)
# Configure generators
train_gen = train_datagen.flow(
    training_images,
    training_labels,
    batch_size=64
)

val_gen = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=64
)

# Train the Model
history = model.fit_generator(
    train_gen,
    epochs=20,
    validation_data=val_gen)

model.evaluate(testing_images, testing_labels, verbose=0)

# Plot the chart for accuracy and loss on both training and validation
%matplotlib inline
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


Exercise 4 - Multi-class classifier
This is the same exercise and notebook as provided here. This button below will take you to the Google Colaboratory environment, in case you would like to use it to follow along with the course videos. In order to pass the graded item, you will still need to submit your work via the Coursera-hosted Jupyter Notebook.
----------------------------------------------------------------------------------
Now that you've explored the concepts behind going from binary classification to multi class classification, it's time for another Exercise. In this one you'll use the Sign Language dataset from https://www.kaggle.com/datamunge/sign-language-mnist, and attempt to build a multi-class classifier to recognize sign language!
Let's build a multi-class classifier to recognize sign language!

This course uses a third-party tool, Exercise 4 - Multi-class classifier, to enhance your learning experience. No personal information will be shared with the tool.

From <https://www.coursera.org/learn/convolutional-neural-networks-tensorflow/ungradedLti/PgmUj/exercise-4-multi-class-classifier> 

Word encoding:-



Convert text to seq



If word is not present
Seq will not have it




Oov is out of vocab word

Padding








Tokenizer.word_index dictionary of numbers and words




Convert json to list


The following is the public domain dataset based on sarcasm, as depicted in the previous video.
The link is provided here for your convenience:
Sarcasm in News Headlines Dataset by Rishabh Misra

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/kyEkF/news-headlines-dataset-for-sarcasm-detection> 

Exercise 1- Explore the BBC news archive
For this exercise you’ll get the BBC text archive. Your job will be to tokenize the dataset, removing common stopwords. A great source of these stop words can be found here.

This course uses a third-party tool, Exercise 1- Explore the BBC news archive, to enhance your learning experience. No personal information will be shared with the tool.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/ungradedLti/LEiBd/exercise-1-explore-the-bbc-news-archive> 

Please find the link to he IMDB reviews dataset here
You will find here 50,000 movie reviews which are classified as positive of negative.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/Z8ypr/imdb-reviews-dataset> 




Embedding layer


Global average pooling is an important technique time saving



The pervious video referred to a colab environment you can practice one.
Here is the link

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/oHNdd/try-it-yourself> 




##check the embedding layer weights



Reverse word index:-



Write vector



Projector .tensorflow.org


For tensorflow 1.13.1  




Sarcasm dataset:-






Vocab size and dimension are hyperparameters in the word embedding


Tkenized datasets reference 
Please find here the datasets url.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/r1kev/tensoflow-datasets> 


Please find the url here

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/C2ocn/subwords-text-encoder> 



Exercise 2- BBC news archive
This week you will build on last week’s exercise where you tokenized words from the BBC news reports dataset. This dataset contains articles that are classified into a number of different categories. See if you can design a neural network that can be trained on this dataset to accurately determine what words determine what category. Create the vecs.tsv and meta.tsv files and load them into the embedding projector.

This course uses a third-party tool, Exercise 2- BBC news archive, to enhance your learning experience. No personal information will be shared with the tool.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/ungradedLti/92HAe/exercise-2-bbc-news-archive> 



RNN:-o/p of prev is fed to current node

Here is the link to Andrew's course on sequence modeling.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/1sxLT/link-to-andrews-sequence-modeling-course> 



LSTM :--cell state
Please find here a link to more information on LSTMs (Long Short Term Memory cells) by Andrew.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/0wa7l/more-info-on-lstms> 








Stacking of LSTM



CNNN layer







We've created a number of notebooks for you to explore the different types of sequence model.
Spend some time going through these to see how they work, and what the impact of different layer types are on training for classification.
IMDB Subwords 8K with Single Layer LSTM
IMDB Subwords 8K with Multi Layer LSTM
IMDB Subwords 8K with 1D Convolutional Layer
Sarcasm with Bidirectional LSTM
Sarcasm with 1D Convolutional Layer
IMDB Reviews with GRU (and optional LSTM and Conv1D)

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/TAAsf/exploring-different-sequence-models> 



Exercise 3- Exploring overfitting in NLP
When looking at a number of different types of layer for text classification this week you saw many examples of overfitting -- with one of the major reasons for the overfitting being that your training dataset was quite small, and with a small number of words. Embeddings derived from this may be over generalized also. So for this week’s exercise you’re going to train on a large dataset, as well as using transfer learning of an existing set of embeddings.
The dataset is from:  https://www.kaggle.com/kazanova/sentiment140. I’ve cleaned it up a little, in particular to make the file encoding work with Python CSV reader.
The embeddings that you will transfer learn from are called the GloVe, also known as Global Vectors for Word Representation, available at: https://nlp.stanford.edu/projects/glove/

This course uses a third-party tool, Exercise 3- Exploring overfitting in NLP, to enhance your learning experience. No personal information will be shared with the tool.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/ungradedLti/zz2wv/exercise-3-exploring-overfitting-in-nlp> 

Please find the link to Laurences generated poetry here.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/dmxZ0/link-to-laurences-poetry> 



Find the link to generating text using a character-based RNN here.

From <https://www.coursera.org/learn/natural-language-processing-tensorflow/supplement/iGKsf/link-to-generating-text-using-a-character-based-rnn> 



Time series :-

Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/A5K7c/introduction-to-time-series-notebook> 

Metrics:-

Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/e2BXk/forecasting-notebook> 

ML on Time series




Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/I9kZS/preparing-features-and-labels-notebook> 

Single layer NN

Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/grB9v/single-layer-neural-network-notebook> 





Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/joICF/deep-neural-network-notebook> 







For seq output





Lambda layers:-->




Huber loss

Please find the Wikipedia page here.

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/umYJD/more-info-on-huber-loss> 





Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/Vv3If/rnn-notebook> 

LSTM:->>


LSTM lesson
Please find the link to the LSTM lesson here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/Ls0UU/link-to-the-lstm-lesson> 







LSTM notebook:-
Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/EtewX/lstm-notebook> 



More information on CNNs can be found on Andrews course within the Deep Learning Specialization.
If you are curious about the content and would like to learn more, here is the link to the course.

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/DM4fi/convolutional-neural-networks-course> 



Bidirectional LSTM:->


Please find more information here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/4aMo3/more-on-batch-sizing> 

Please find the link to the notebook here

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/x54Iv/lstm-notebook> 



Sunspots notebook

Please find the link to the notebook here
For fun, there's also a version of the notebook that uses only DNN here.

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/supplement/4v7Xh/sunspots-notebook> 

Exercise 4 - Sunspots
This week you moved away from synthetic data to do a real-world prediction -- sunspots. You loaded data from CSV and built models to use it. For this week’s exercise, you’ll use a dataset from Jason Brownlee, author of the amazing MachineLearningMastery.com site and who has shared lots of datasets at https://github.com/jbrownlee/Datasets. It’s a dataset of daily minimum temperatures in the city of Melbourne, Australia measured from 1981 to 1990.  Your task is to download the dataset, parse the CSV, create a time series and build a prediction model from it. Your model should have an MAE of less than 2, and as you can see in the output, mine had 1.78. I’m sure you can beat that! :)

From <https://www.coursera.org/learn/tensorflow-sequences-time-series-and-prediction/ungradedLti/sFRBW/exercise-4-sunspots> 


![image](https://github.com/user-attachments/assets/973d7eaa-18e9-41d8-a9de-2cd87f9da8e8)
