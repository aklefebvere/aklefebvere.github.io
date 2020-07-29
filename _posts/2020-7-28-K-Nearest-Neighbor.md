---
layout: post
title: K-Nearest Neighbor For Dummies
subtitle: Data Science Blog #3
---

# Introduction
One of the best algorithms to start with when beginning to learn about machine learning is the K-nearest neighbor model or K-NN. K-nearest neighbor is considered by many people the simplest algorithms someone can learn. In this blog, I will be discussing the following questions:
  * What is a K-nearest neighbor model?
  * How does it work?
  * How do I create a K-nearest neighbor model?
  * What are the use cases for a K-nearest neighbor model?

By the end of the day, you should be able to answer to yourself all these questions and even hopefully code your very own K-nearest neighbor model.

# What is a K-nearest neighbor model?
K-nearest neighbor model is a machine learning model used most of the time for classifying new data. K-NN is good at classifying new data because it is a supervised machine learning algorithm which means that it requires labeling for each row of data. When training data is passed into a K-NN model, the training data does not get modifyed, it's simply stored into the model for it to be used for predictions. Now we know what is a K-nearest neighbor model is but how does it work?

# How does the algorithm work?

![test](/img/KNN_graph_2.png)

K-nearest neighbor works by getting the euclidian distance from a test point to the k closest points to the test point. K-NN specifies how many neighbors to pick by using the letter k and k is defined by the user of the model. In the data visualization above, the red point is a data point from the test dataset and since k is set to five, it will get the 5 closest points to the test point which is what the red lines are pointing to. Once the k closest points have been identified and selected, a voting process starts to determine what class to identify the test point. Since all the points that were selected were from the Iris-setosa class, the red test point predicted class is the Iris-setosa class. If for example, there was three Iris-setosa points and two Iris-virginica points, the predicted class would be Iris-setosa. Iris-setosa was the predicted class because there was more points selected for Iris-setosa then there was for Iris-virginica which is all determined by the algorithm's voting process. To prevent having ties in the voting process (ex: k=2, 1 Iris-setosa, 1 Iris-virginica), it is best practice to pick an odd number for k so that there cannot be a tie in the classes selected. Now that we know what is a K-nearest neighbor model is and how the algorithm works, we can now begin coding the model from scratch.

# How do I create a K-nearest neighbor model?
Before we start coding, lets plan out what we need to code:
  * A class to hold our built K-NN model
    * two variables two hold our X_train and y_train
    * an attribute to set the value of k
  * A class method to calculate the euclidian distance from two points
    * we can use the numpy library to accomplish this
  * A class method to fit our model on the class
    * two parameters for X_train and y_train and set those passed in parameters as the attributes of the clas
  * A class method to predict the class(es) of the test data
    * Get the distance of one test points to all test points
    * pick the k closest points to the test point
    * Get the classes of the k closest points
    * Go through the voting process and determine whats the predicted class
    * If there was multiple test rows plugged in, re-run the above steps
    * return all the predicted classes
    
Now that we have our plan, lets start executing our plan into actual python code. Lets start by creating the K-NN class.
```python
class K_NN:
    X_train = None
    y_train = None
    def __init__(self, K):
        self.K = K
```
We have created a class called K_NN with two variables called X_train and y_train and a initualizer method with the parameter K. The two variables will hold our training data and the parameter K will hold the number of nearest neighbors we are going to collect. Now lets create the class method to calculate the euclidian distance from two points.
```python
def euclidean_distance(self, row1, row2):
        return np.linalg.norm(row1-row2)
```
In the method, row1 will take in an individual row of the training data and row2 will take in a individual test row from the test data. np.linalg.norm(row1-row2) is the actual calculation of the euclidian distance between two vectors. When we say .norm, we are telling linalg to return the distance of the two vectors. If you wanted to write that portion by hand, you could also write something like this ```np.sqrt(np.sum((v1 - v2) ** 2))``` which does the same thing as the numpy method. Now lets create the fit method of our class.
```python
def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train
```
This simply sets the passed in training data as the training data for the class object. This will be used in the predict method. Finally, lets write up the predict method. I'm going to break it down into multiple steps since it's our biggest method.
```python
 def predict(self, pred):
        predictions = []
        for p in pred:
            distances = []
            for i, v in enumerate(self.X_train):
                distance = self.euclidean_distance(v, p)
                distances.append([distance, i])
```
We create an empty list called predictions that will hold all our predicted classes. ```for p in pred``` will iterate through all the plugged in test vectors and the distances list will hold all of our distances for that indiviudal test vector. ```for i, v in enumerate(self.X_train)``` will iterate through all the vectors in our training data and then we will get the distance of an individual training vector to our individual test vector. We will then append the distance of an individual training vector to our indiviudal test vector to the distances list we created with index of the training vector and we will do this for all vectors in the training data. Next lets sort the distances and pick out k distances.
```python
sorted_distances = sorted(distances)
k_distances = sorted_distances[:self.K]
predict = [self.y_train[i[1]] for i in k_distances]
```
Now that we have all the distances from one test vector to all vectors in the training data, we can sort the distances from least to greatest ```sorted(distances)``` and then get the k closest points to the test vector ```sorted_distances[:self.k]```. Since we saved the index of the training vector, we can then find what was the classification for each k points by doing ```[self.y_train[i[1]] for i in k_distances]```. The last step for the predict method is to select the class for a test vector.
```python
result = max(set(predict), key = predict.count)
predictions.append(result)
return predictions
```
```max(set(predict), key= predict.count)``` will get the class that occured the most in the list and then that will get appended into the preditions list. If more than one test vector was passed in at once, it will go back up to the ```for p in pred``` for loop and do everything again until there is no more test vectors to iterate. Once everything is finished, it will return the predicted classes for every test vector the user plugged into the class object.

Now that we have all of our pieces of the class completed, we can now put all of this into one big class and create the complete K-nearest neighbor class. I will add comments for each line to briefly describe what I explained above.
```python
import pandas as pd
import numpy as np

class K_NN:
    X_train = None
    y_train = None
    def __init__(self, K):
        self.K = K

    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, row1, row2):
        return np.linalg.norm(row1-row2)

    def predict(self, pred):
        # empty list to hold each prediction
        predictions = []
        for p in pred:
            # empty list to hold distances for a specific test row
            distances = []
            # for loop to iterate through every row in the training set
            for i, v in enumerate(self.X_train):
                # calculate the euclidean distance between a training row
                # and test row
                distance = self.euclidean_distance(v, p)
                # append the distance to the distances list
                distances.append([distance, i])
            # sort the distances from least to greatest
            sorted_distances = sorted(distances)
            # take only smallest k distances
            k_distances = sorted_distances[:self.K]
            # Get the predicted classification
            predict = [self.y_train[i[1]] for i in k_distances]
            # Get the most frequent predicted element from predict
            result = max(set(predict), key = predict.count)
            # Append the result to the predictions list
            predictions.append(result)

        return predictions
```
Congratulations! You have coded your very own K-nearest neighbor model! Now that this is complete, we can test it out with actual data.
