# Hafsa's Projects

# [Project : Distributed computing](https://github.com/aoutir/Distributed-Algorithms) 

* Implement in Java building blocks of distributed systems typically reliable broadcast. 
* The abstractions implemented are: perfect links, best effort broadcast, uniform reliable broadcast, FiFo broadcast and Lattice agreement. 

# [Project  : Explore the impact of Elon Musk’s quotations on the stock market using tesla stock market](https://github.com/epfl-ada/ada-2021-project-noname)

* Using quotes retrieved from Quotebank and Tesla’s stock price we build a predictive model to forecast daily stock return of Tesla.
* Analysing Tesla's stock price from 2015 until 2020
* Preprocessing of quotations: Tokenization, lemmatization, removing stop words and punctuation. 
* Sentiment Analysis of Elon Musk's quotations using two pretrained models Vader and Textblob. 
* Perform TF-IDF on Elon musk quotes to add features to our model: which was proven to be efficent. 
* Compare different predictive models to find the most accurate one for our study for instance linear regression, gradient boosting regressor, SVM and MLP classifier.
* Described the project in a data story [here](https://aoutir.github.io/).

# [Project : Road segmentation challenge](https://github.com/aoutir/Project_Machine_Learning)

* Train a classifier to segment roads given a set of satellite images from google maps using DeepLabV3 Resnet-101 achieved an F1 score of 0.920 on the test set. 
* Since the provided dataset was small, we had to perform data augmentation techniques on the images. Torchvision was used allowing a fine-grained control of the transformation pipeline. 
* Model selection (U-net, DeepLabV3, LR-ASPP mobileNetV3 Large) and hyperparameters tuning using crossvalidation, Grid search/Random search. 



# [Project : Predicting stock indices using neural networks](https://github.com/aoutir/Project_Deep_learning)

* Using convolutional neural network to predict whether following day's close price of an index will be higher or lower than the current day given a dataset of DJI, NASDAQ 100, New work stock exchange, RUSSEL and SP. 
* Improved CNNpred a previously implemented neural network of the [paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417419301915). 
* Run experiments on different models CNNpred, stackedCNNpred, LSTM, CNN-LSTM. 
* Achived an F1 score of 0.53 with the stackedCNNpred which was quite impressive since the training set was small.


# Project : Data analytics for short term forecasting of electrical power with the lab LCA2 at EPFL 
* The aim of the project is to maximize the consumption of electrical power by accurately forecasting power consumption using historical data measurements. There are two types of power consumptions, and we believe that if the same conditions are reproduced, the power magnitude will behave similarly. To achieve this, a set of influential features has been identified, such as time and day, and  cluster the historical power measurements according to these features. The power magnitudes of each cluster is represented in a histogram and it used to predict power consumption for new days. The goal is to discover the optimal set of features for clustering that provides the most accurate predictions. By using this approach, the goal is to maximize the power consumption while minimizing waste and cost.
* Exploratory data analysis in order to get insights about the historical measurements of power generation.
* Studying the optimal classification criterion for power magnitudes using clustering techniques: Kmeans and GMM.
* Analyse the influence of features using statistical tests Anova correlation and Pearson correlation coefficient.  
* Unfortunately I can't make the code of this project public since it was done with the help of the Lab. 


# [Project : DECIDE ](https://github.com/aoutir/DECIDE)

* DECIDE() is a software application implemented in Java that generates a boolean signal determining whether an interceptor should be launched based on input radar tracking information. This radar tracking information is available at the instant the function is called.

# [Project : Emulator of gameboy](https://github.com/aoutir/Emulator_gameboy)

* Implemented a nintendo emulator in C, the main components of the gameboy were implemented for instance cpu, bus controller, ROM, DRAM and many more.

# [Project : Grid game](https://github.com/aoutir/game_java)

* Implemented a grid game in java where the player will have to solve puzzles for example: (a) break a rock with
an object to find, (b) and (c) finding one's way in a maze with or without a field
restricted vision, (d) activate all signals by stepping on them, (e) find the correct
combination of levers or (f) work your way through by pushing rocks to gain access to
useful resources or other game levels.

# [Project : Tangible game](https://github.com/aoutir/Project_computer_vision) 

* 3D game contrallable using webcame implemented in Processing. 

# [Project : Jass Game](https://github.com/aoutir/game_java)

* Implemented the famous card game Jass in Java with the graphical interface.

