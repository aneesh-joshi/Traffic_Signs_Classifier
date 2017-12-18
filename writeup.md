# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/image1.png "Visualization 1"
[image2]: ./images/image2.png "Visualization 2"
[image3]: ./images/image3.png "Test Images 1"
[image4]: ./images/image4.png "Test Images 2"
[image5]: ./images/image5.png "Test Images 3"
[image6]: ./images/image6.png "Top5 1"
[image7]: ./images/image7.png "Top5 2"
[image8]: ./images/image8.png "Top5 3"
[image9]: ./images/image9.png "Top5 4"
[image10]: ./images/image10.png "Top5 5"
[image11]: ./images/image11.png "Top5 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/aneesh-joshi/Traffic_Signs_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the used manipulations of numpy arrays to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

Here, I use a histogram to show the distribution of classes within the different data splits of
* Training
* Testing
* Validation

![alt text][image1]


Here, I have explored the different traffic signs that exist:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Initially, I tried to normalize the data at the beginning. But this required a high amount of RAM which I didn't have. After some consideration, I devised the solution of normalising on a batch basis.
I used a simple normalizing scheme of `x = x / 255` 

I decided against grayscaling the data as I felt that the colors of the traffic signs play a big role in deciphering the class that they belong to.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout				| keep probablity of 0.6						|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 					|
| Convolution 3x3	    | 1x1 stride, outputs 10x10x16       			|
| RELU 					|												|
| Dropout				| keep probablity of 0.6						|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 					|
| Flatten				| 400 values									|
|						|												|
| Fully connected		| output: 400        							|
| RELU					|												|
| Dropout				| keep probablity of 0.5						|
|						|												|
| Fully connected		| output: 120        							|
| RELU 					|												|
| Dropout				| keep probablity of 0.5						|
|						|												|
| Fully connected		| output: 84        							|
| RELU 					|												|
| Dropout				| keep probablity of 0.5						|
|						|												|
| Fully connected		| output: 43        							|
| Softmax				|         									    |

 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used an Adam Optimizer to minimize the cross entropy classification loss.
A useful tweak which I made was to set the learning rate as a placehodler. This allowed me to update the learning rate as the training proceeded without updating the 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 95.7%
* test set accuracy of 94.4%

I struggled with this problem for a very long time.
After setting up the initial model, the validation loss did not drop for below 0.054
It took me a lot of iterations before I found the reason.
I had decided not to shuffle the data as it required a lot of RAM (which I don't have)
Turns out, shuffling data is extremely important. It lead to drastic improvements in the results.

I also noticed that setting a high learning rate leads to no learning. So, I started with a small rate.

With these in place, I managed to get a validation accuracy of 90%
To go further, I decided to set a dropout on the convolutions as well as fully connected layers.
This boosted the validation accuracy of 95%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4] ![alt text][image5] 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).


I faced an interesting problem while solving this one.
The images I had found were of arbitrary dimensions. So, I had to resize them to be 32x32.
Naively, I resized with an antialias setting. This resulted in 0% accuracy.
On inspecting the training images, I realised that they weren't antialiased.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work   									| 
| Stop   	  			| Turn Left 									|
| Stop   	  			| Turn Left 									|
| 70 km/h				| 70 km/h										|
| 30 km/h	      		| 30 km/h					 				    |
| Ahead Only			| Ahead Only      							    |


The model was able to correctly guess 4 of the 6 traffic signs, which gives an accuracy of 66%. Considering that two of the images were of the same sign, it's more like 4/5 which is 80%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


![alt text][image6]
The model is very sure about this sign (86%).


Unfortunately, for both the stop signs the model is sure of turn left and stop doesn't even appear in the top 5 probablities. This might have something to do with backgrounds which the network has picked up on.

![alt text][image7]
![alt text][image8]


Here, the network is realtively highly confident of the sign (56%)
Interestingly, the 3 of the other next highest probablities are speed limits.

![alt text][image9]


Similarly for this speed limit.

![alt text][image10]


Here too, there is a strong probablity towards the correct sign(51%) with the next competition of 47% from the yield sign.

![alt text][image11]
