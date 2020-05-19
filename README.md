I consider myself a data science enthusiast who is always to eager to learn new topics and implement new ideas. I also write about data science, machine learning, deep learning and statistics. Below is a list of the projects I have done and some of the stories from my blog on Medium.

# Projects

### Image Classification with Deep Learning

**Motivation**: Computer vision is an highly important field in data science with many applications from self-driving cars to cancer diagnosis. Convolutional neural networks (CNNs) are commonly used for computer vision and image classification tasks. I implemented a CNN using Keras to perform a binary classification task. I tried to explain the concepts of each step in a convolutional neural network and the theory behind them.

**Data**: The images are taken from [Caltech101 dataset](http://www.vision.caltech.edu/Image_Datasets/Caltech101/)

**Achievements**:
* Preprocess the images using ImageDataGenerator of Keras. ImageDataGenerator generates batches of tensor image data with real-time data augmentation by applying random selections and transformations (such as rotating and shifting) in batches. Data augmentation increases the diversity of the dataset and thus helps to get more accurate results and also prevents the model from overfitting.
* Build a CNN model with convolution and pooling layes as well as a flattenning and a dense layer.
* The model achieved 99% accuracy on training set and 98.1% accuracy on test set.

[Model accuracy](https://github.com/SonerYldrm/Image-Classification-with-CNNs/blob/master/Model_accuracy.png)

[GitHub repo of the project](https://github.com/SonerYldrm/Image-Classification-with-CNNs)

### Cryptocurrency Prediction with Deep Learning

Motivation: Although the first redistributed cryptocurrency (bitcoin) was created in 2009, the idea of digital money arised in 1980s. In the recent years, cryptocurrencies have gained tremendeous popularity. As traditional currencies, the value of cryptocurrencies are changing in time. Using the historical data, I will implement a recurrent neural netwok using LSTM (Long short-term memory) layers to predict the trend of value of a cryptocurrency in the future.

Data: There is a huge dataset about [cryptocurrency market prices](https://www.kaggle.com/jessevent/all-crypto-currencies) on Kaggle. I only used a part of it which is historical price data of litecoin.

Achievements:
* Preprocess the data and convert it to a format that can be with to LSTM layer
* Build a deep learning model with Keras that includes LSTM layers and a dense layer
* The model achieved to reduce the loss below 0.002 on training set and also predict the trend on the test set.

![Test Set- Actual vs Real](https://github.com/SonerYldrm/Currency-Prediction-with-RNN/blob/master/Test_set_prediction.png)

How to Improve:
We can build a more robust and accurate model by collecting more data. We can also try to adjust number of nodes in a layer or add additional LSTM layers. We can also try to increase the number of timesteps which was 90 in our model.

[GitHub repo of the project](https://github.com/SonerYldrm/Churn-Prediction)

### Churn Prediction

Motivation: Churn prediction is common use case in machine learning domain. It is very critical for business to have an idea about why and when customers are likely to churn (i.e. leave the company). Having a robust and accurate churn prediction model helps businesses to take actions to prevent customers from leaving the company.

Data: I used the telco customer churn dataset available on Kaggle. The dataset includes 20 featuures (independent variables) and 1 target (dependent) variable for 7043 customers. 

Achievements:
* With an extensive exploratory data analysis process, I understand the characteristics of features as well as the relationship among them. Then, eliminated the redundant features.
* Encode categorical features so that they can be used as input to a machine learning model. Also, normalized the numerical values.
* Implemented two models:
1. Ridge classifier: Achieved 76.1% accuracy on test set
2. Random forests: Initially achievend 84.2% accuracy but I managed to increase the accuracy to 90% with hyperparameter tuning.

How to Improve:
The fuel of machine learning models is data so if we can collect more data, it is always helpful in improving the model. We can also try a wider range of parameters in GridSearchCV because a little adjustment in a parameter may slighlty increase the model.

[GitHub repo of the project](https://github.com/SonerYldrm/Churn-Prediction)


# Blog

## Machine Learning Algorithms

### Supervised Learning

* [Support Vector Machines](https://towardsdatascience.com/support-vector-machine-explained-8d75fe8738fd?source=friends_link&sk=677804e88752a496a154ec74bc6a04ab)

* [Decision Trees and Random Forests](https://towardsdatascience.com/decision-tree-and-random-forest-explained-8d20ddabc9dd?source=friends_link&sk=2312f2149c10f0804b57bd73a8942004)

* [Naive Bayes Classifiers](https://towardsdatascience.com/naive-bayes-classifier-explained-50f9723571ed?source=friends_link&sk=dff592652eb7f6589997df67b94f3d5e)

* [Gradient Boosted Decision Trees](https://towardsdatascience.com/gradient-boosted-decision-trees-explained-9259bd8205af?source=friends_link&sk=69bae99ff05784e2f18412a30e4ee4c1)

* [Logistic Regression](https://towardsdatascience.com/logistic-regression-explained-593e9ddb7c6c?source=friends_link&sk=9c80aae75268c7ef88c488fa6949d3f2)

* [K-Nearest Neighbors (kNN)](https://towardsdatascience.com/k-nearest-neighbors-knn-explained-cbc31849a7e3?source=friends_link&sk=526badeb56f557074d17444b4a1b1b12)

### Unsupervised Learning

* [K-Means Clustering](https://towardsdatascience.com/k-means-clustering-explained-4528df86a120?source=friends_link&sk=4c8c67dd0f3702b4ecd5bd435e82be2a)

* [Hierarchical Clustering](https://towardsdatascience.com/hierarchical-clustering-explained-e58d2f936323?source=friends_link&sk=0dc952162cb32fd1d666488869b40998)

* [DBSCAN Clustering](https://towardsdatascience.com/dbscan-clustering-explained-97556a2ad556?source=friends_link&sk=34729aecd0a0797832a686515ddcb1e3)

* [Principal Component Analysis](https://towardsdatascience.com/principal-component-analysis-explained-d404c34d76e7?source=friends_link&sk=87fcb241b63ad1d06f19ec032fde61f3)


