# Soner Yıldırım
## Data Science Portfolio

## Projects

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


