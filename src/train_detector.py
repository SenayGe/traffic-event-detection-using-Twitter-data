from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pandas as pd
import seaborn as sns
import models
from filter_tweets import database_creation
from utils import *


def train_detector (model = "MLP" , with_tweets = True, tweet_lifetime = 'avg'):
    data_loader = database_creation(lifetime = 287) # You can change lifetime here

    if model == "MLP":

        batch_size=16
        epochs =200
        feature_size = None
        data_source = None
        # Definning our trarget variables and predictors

        if (with_tweets):
            TargetVariable=['Class']
            Predictors = [ 'SensorsID', 'Speed', 'Feature1', 'Feature2', 'Feature3', 'Feature4',
            'Feature5', 'Feature6', 'tweet_lifetime', 'tweet_binary'] 
            feature_size = 10
            data_source = "Sensors and Twitter"
        else:
            TargetVariable=['Class']
            TargetVariable=['Class']
            Predictors=['SensorsID', 'Speed', 'Feature1', 'Feature2', 'Feature3', 'Feature4',
                        'Feature5', 'Feature6'] #'tweet_lifetime', 'tweet_binary'
            feature_size = 8
            data_source = "Sensors only"

        X = data_loader[Predictors].values  # Sensors data only
        y = data_loader[TargetVariable].values

        # Standardizing the values of X
        X, PredictorScalerFit = z_score(X) # Z-score normalization

        # Splitting the data into training and testing set
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

        # Creating the MLP model
        MLP_model = models.mlp_model(feature_size)

        # Training/Fitting the MLP model
        training_history = MLP_model.fit(xTrain,yTrain, batch_size=batch_size , epochs=epochs, verbose=1)

        # Plotting the training results
        plot_training_results(training_history)

        # Testing the our training model and generating prediction
        Predictions=MLP_model.predict(xTest)
        
        # Scaling the test data back to original scale
        Test_Data=PredictorScalerFit.inverse_transform(xTest)
        
        # Generating a data frame for analyzing the test data
        TestingData=pd.DataFrame(data=Test_Data, columns=Predictors)
        TestingData['TrafficEvent']=yTest
        TestingData['PredictedEventProb']=Predictions

        # Generating predictions on the testing data by applying probability threshold
        TestingData['PredictedTrafficEvent']=TestingData['PredictedEventProb'].apply(probThreshold)

        # Printing testing results
        print(f'\n######### Testing Accuracy Results - {data_source} #########')
        print(metrics.classification_report(TestingData['TrafficEvent'], TestingData['PredictedTrafficEvent'], digits = 4))
        print(metrics.confusion_matrix(TestingData['TrafficEvent'], TestingData['PredictedTrafficEvent']))
        matrix_confusion = confusion_matrix(TestingData['TrafficEvent'], TestingData['PredictedTrafficEvent'])

        # Plotting the confusion matrix
        sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)

    elif model == "CNN":
        verbose, epochs, batch_size = 0, 50, 16
        feature_size = 10
        TargetVariable=['Class']
        Predictors = [ 'SensorsID', 'Speed', 'Feature1', 'Feature2', 'Feature3', 'Feature4',
            'Feature5', 'Feature6', 'tweet_lifetime', 'tweet_binary'] 
        X = data_loader[Predictors].values  # Sensors data only
        y = data_loader[TargetVariable].values
        # Splitting the data into training and testing set
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

        # Reshaping the data into 3dim
        xTrain = xTrain.reshape(-1,1 , 10) # shape of X is 4D, (1, 2, 2, 1)
        xTest = xTest.reshape(-1, 1, 10)
        n_timesteps, n_features, n_outputs = xTrain.shape[1], xTrain.shape[2], 1

        # Creating the CNN model
        CNN_model = models.cnn_model(feature_size)

        CNN_model.fit(xTrain, yTrain, epochs=epochs, batch_size=batch_size, verbose=1)
        # evaluate model
        _, accuracy = CNN_model.evaluate(xTest, yTest, batch_size=batch_size, verbose=0)

        print ('\n######### Testing Accuracy Results - CNN  with Senor data and Tweets #########')
        print('Accuracy: %.2f' % (accuracy*100))