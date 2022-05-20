from filter_tweets import database_creation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import pandas as pd
from utils import *
from models import *

from train_detector import *
from ML_training import *

def main ():


    # -------Training our traffic event detector with MLP Model  -------------

    # Using Sensor data only
    train_detector (model = "MLP" , with_tweets = False)

    # Using Sensor and Twitter data
    train_detector (model = "MLP" , with_tweets = True)

    # -------Training our traffic event detector with CNN Model  -------------

    # Using Sensor and Twitter data
    train_detector (model = "CNN" , with_tweets = True)


    # -------Training our traffic event detector with MLL Model  -------------
    ''''
     If you want to train the model with MLL Model, you need to uncomment the following lines
    '''
    # remove_sensor = False
    # remove_description = True
    # sample = 'Oversampling'
    # train_knn(remove_description,remove_sensor,sample)
    # train_decisionTree(remove_description,remove_sensor,sample)
    # train_RandomForest(remove_description,remove_sensor,sample)
    # train_GBS(remove_description,remove_sensor,sample)
    # train_XGB(remove_description,remove_sensor,sample)

if __name__ == "__main__":
    main()