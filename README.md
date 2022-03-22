# Hafsa_Portfolio

# [Project 1: Explore the impact of Elon Musk’s quotations on the stock market using tesla stock market](https://github.com/epfl-ada/ada-2021-project-noname)

* Using quotes retrieved from Quotebank and Tesla’s stock price we build a predictive model to forecast daily stock return of Tesla.
* Analysing Tesla's stock price from 2015 until 2020
* Preprocessing of quotations: Tokenization, lemmatization, removing stop words and punctuation. 
* Sentiment Analysis of Elon Musk's quotations using two pretrained models Vader and Textblob. 
* Perform TF-IDF on Elon musk quotes to add features to our model: which was proven to be efficent. 
* Compare different predictive models to find the most accurate one for our study for instance linear regression, gradient boosting regressor, SVM and MLP classifier.


# [Project 2: Road segmentation challenge](https://github.com/aoutir/Project_Machine_Learning)

* Train a classifier to segment roads given a set of satellite images from google maps using DeepLabV3 Resnet-101 achieved an F1 score of 0.920 on the test set. 
* Since the provided dataset was small, we had to perform data augmentation techniques on the images. Torchvision was used allowing a fine-grained control of the transformation pipeline. 
* Model selection (U-net, DeepLabV3, LR-ASPP mobileNetV3 Large) and hyperparameters tuning using crossvalidation, Grid search/Random search. 



# [Project 3: Predicting stock indices using neural networks](https://github.com/aoutir/Project_Deep_learning)

* Using convolutional neural network to predict whether following day's close price of an index will be higher or lower than the current day given a dataset of DJI, NASDAQ 100, New work stock exchange, RUSSEL and SP. 
* Improved CNNpred a previously implemented neural network of the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417419301915). 
* Run experiments on different models CNNpred, stackedCNNpred, LSTM, CNN-LSTM. 
* Achived an F1 score of 0.53 with the stackedCNNpred wich was quite impressive since the training set was small.



