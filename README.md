# Hafsa_Portfolio

# [Project 1: Explore the impact of Elon Musk’s quotations on the stock market using tesla stock market](https://github.com/epfl-ada/ada-2021-project-noname)

* Using quotes retrieved from Quotebank and Tesla’s stock price we build a predictive model to forecast daily stock return of Tesla.
* Analysing Tesla's stock price from 2015 until 2020
* Preprocessing of quotations: Tokenization, lemmatization, removing stop words and punctuation. 
* Sentiment Analysis of Elon Musk's quotations using two pretrained models Vader and Textblob. 
* Perform TF-IDF on Elon musk quotes to add features to our model: which was proven to be efficent. 
* Compare different predictive models to find the most accurate one for our study for instance linear regression, gradient boosting regressor, SVM and MLP classifier.


# [Project 2: Road segmentation challenge](https://github.com/CS-433/ml-project-2-hse)

* Train a classifier to segment roads given a set of satellite images from google maps using DeepLabV3 Resnet-101 achieved an F1 score of 0.920 on the test set. 
* Since the provided dataset was small we had to perform data augmentation techniques on the images standard options from the PyTorch library : torchvision were used, with Functional transforms allowing a fine-grained control of the transformation pipeline. 
* To choose the most suitable predictive model for segmentation we compared different models to our baseline model which is a simple convolutional neural network. (U-net, DeepLabV3, LR-ASPP mobileNetV3 Large) 
* We trained all the models on our training set by pytorch on TorchVision.models.segmentation.
* We performed hyperparameters tuning using gridsearh and random-search.



