Image Classification Using Stochastic Gradient Descent and KNeighborsClassifier
==========================

![Learn Machine learning](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/header.png)

## Introduction

* A machine learning classification model that is trained on the MNIST images dataset from the ML data.org repository. 
* We are doing supervised learning here and our aim is to do image classification and noise reduction 
* During our journey we'll understand the important tools needed to develop a powerful ML model
* Our aim is to play with tools like Stochastic Gradient Descent, KNeighbors classifier, confusion matrix, Precision, Recall, ROC curves, Area under ROC, cross validation and GridSearchCV to reach our goal.
* We'll evaluate the performance of each of our classifier using Precision scores, Recall scores, and also tune hyper parameters to further optimize our model
* We'll validate our predictions against our test dataset and conclude our learnings


### 1. We'll learn how to analyze mis-interpreted digits
![Misinterpreted 3s and 5s](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/misinterpreted_3s_and_5s.png)

### 2. To find where our classifier is making mistakes we'll plot the confusion matrix to pin down the error 
![Confusion Matrix](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/confusion_matrix.png)

### 3. To optimize our classifier we'll plot Precision-Recall Vs Threshold curves.
![Precision-Recall Vs Threshold](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/Precision_Recall_Vs_Threshold.png)

### 4. To compare the performance of our models we'll plot their ROC AUC
![Roc curve](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/ROC_curve_SGD_vs_RandomForest.png)

### 5. We will train a KNeighbors classifier to do noise reduction from corrputed images.
![Roc curve](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/noisy_digit_example_plot.png)

### 6. Conclusion
At the end of our project we'll be able to derive insights to present to our team. 

![Presentation](https://raw.githubusercontent.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset/master/images/presentation.jpg)

## To do an end-to-end Machine Learning project we need to do the following steps

1. Understand the requirements of the business.
2. Acquire the dataset.
3. Visualize the data to understand it better and develop our intuition.
4. Pre-process the data to make it ready to feed to our ML model.
5. Try various models and train them. Select one that we find best.
6. Fine-tune our model by tuning hyper-parameters
7. Present our solution to the team.
8. Launch, monitor, and maintain our system.


## Installation

First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/gurupratap-matharu/machine-learning-classification-MNIST-dataset


If you are familiar with Python and you know how to install Python libraries, go ahead and install the libraries 

## Python & Required Libraries

You can check which version(s) you have by typing the following commands:

    $ python3 --version  # for Python 3


* If you don't have Python 3, I recommend installing it (Python ≥2.6 should work, but it is deprecated so Python 3 is preferable). 


On Linux, unless you know what you are doing, you should use your system's packaging system. For example, on Debian or Ubuntu, type:

    $ sudo apt-get update
    $ sudo apt-get install python3


## Using Anaconda

When using Anaconda, you can optionally create an isolated Python environment dedicated to this project. This is recommended as it makes it possible to have a different environment for each project (e.g. one for this project), with potentially different libraries and library versions:

    $ conda create -n venv python=3.5 anaconda
    $ conda activate venv

This creates a fresh Python 3.5 environment called `venv` (you can change the name if you want to), and it activates it. This environment contains all the scientific libraries that come with Anaconda. This includes all the libraries we will need (NumPy, Matplotlib, Pandas, Jupyter and a few others), except for TensorFlow, so let's install it:

   
Great! You're all set, you just need to start Jupyter now.

## Starting Jupyter

You can start Jupyter, simply type:

    $ jupyter notebook

This should open up your browser, and you should see Jupyter's tree view, with the contents of the current directory. If your browser does not open automatically, visit [localhost:8888](http://localhost:8888/tree).  

Great! We are all setup to do some data science and machine learning!


## Note of Thanks

I would like to thank Aurélien Géron and his wonderful book Machine Learning with ScikitLearn and TensorFlow. I highly recommend this book.

