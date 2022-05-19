from filter_tweets import database_creation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pandas as pd
from utils import *
from models import *

def main ():

    data_loader = database_creation(lifetime = 287) # You can change lifetime here

    # -------Training Models using Sensor data only -------------

    # Definning our trarget variables and predictors
    TargetVariable=['Class']
    Predictors=['SensorsID', 'Speed', 'Feature1', 'Feature2', 'Feature3', 'Feature4',
                'Feature5', 'Feature6'] #'tweet_lifetime', 'tweet_binary'


    X = data_loader[Predictors].values  # Sensors data only
    y = data_loader[TargetVariable].values

    # Standardizing the values of X
    X, PredictorScalerFit = z_score(X) # Z-score normalization

    # Split the data into training and testing set
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating the MLP model
    MLP_model = mlp_model(8)

    # Training/Fitting the MLP model
    training_history = MLP_model.fit(xTrain,xTest, batch_size=16 , epochs=200, verbose=1)

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
 
    print('\n######### Testing Accuracy Results - Sensor #########')
    print(metrics.classification_report(TestingData['TrafficEvent'], TestingData['PredictedTrafficEvent'], digits = 4))
    print(metrics.confusion_matrix(TestingData['TrafficEvent'], TestingData['PredictedTrafficEvent']))
    matrix_confusion = confusion_matrix(TestingData['TrafficEvent'], TestingData['PredictedTrafficEvent'])
    sns.heatmap(matrix_confusion, square=True, annot=True, cmap='Blues', fmt='d', cbar=False)

    Predictors_twt = [ 'SensorsID', 'Speed', 'Feature1', 'Feature2', 'Feature3', 'Feature4',
                'Feature5', 'Feature6', 'tweet_lifetime', 'tweet_binary'] #'tweet_lifetime', 'tweet_binary'




