# CNN-On-MNIST-Handwriting-Digit

## Introduction
The MNIST Handwriting Digit dataset is a commonly used dataset in computer vision and machine learning, consisting of a set of 70,000 images of handwritten digits, each with a size of 28x28 pixels. The task is to classify each image into one of the 10 possible digit classes (0-9).

![1](http://i.imgur.com/4o8MTiT.png)

Convolutional Neural Networks (CNNs) are a type of deep neural network that are particularly well-suited for image classification tasks like the MNIST dataset. CNNs are able to learn spatial features from the input images by applying convolutional filters to the input data, which helps to capture local patterns and structures in the data.

The CNN architecture for the MNIST dataset typically consists of several convolutional layers with ReLU activation functions, each followed by a max pooling layer. The convolutional layers extract low-level features from the input images, while the max pooling layers reduce the spatial dimensions of the feature maps and help to make the model more robust to small variations in the input data.

After the convolutional layers, the feature maps are flattened and passed through one or more fully connected layers, which perform a non-linear mapping from the input space to the output space. The fully connected layers are typically followed by a softmax activation function to produce a probability distribution over the possible output classes.

During training, the CNN learns to adjust the weights of its neurons to minimize a loss function that measures the difference between the predicted outputs and the true labels. The most commonly used loss function for the MNIST dataset is the sparse categorical cross-entropy loss, which is suitable for multi-class classification problems where the labels are integers.

To prevent overfitting, dropout regularization can be applied to the fully connected layers, which randomly drops out some of the neurons during training. This helps to reduce the model's reliance on specific neurons and makes it more robust to noise in the input data.

Overall, training a CNN on the MNIST Handwriting Digit dataset is a common exercise in deep learning and provides a good introduction to the basic principles of CNNs and image classification.


![2](http://i.imgur.com/kzBAJEa.png)

## Requirements

* Python 2.7
* [Tensorflow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* numpy
* matplotlib
* pandas

## Dataset

* The model is trained on the MNIST dataset downloaded from tf.keras.datasets.mnist.load_data(https://github.com/keras-team/keras/blob/v2.11.0/keras/datasets/mnist.py#L25-L86)

## The neural network used in this model:
<p align="center">
  <img src="https://github.com/BerqiaMouad/softmax_digit_classification/blob/master/NN_model.png">
</p>

### The accuracy of the model is approximatly :

+ On the trainning set = 99.77%
+ On the test set = 99.50 %

### Some informations:
<b>To run directly run the ipynb file step by step </b>
<br/><br/>

