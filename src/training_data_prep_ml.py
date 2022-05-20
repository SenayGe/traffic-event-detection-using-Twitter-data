from random import shuffle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split	
import pandas as pd
from filter_tweets import database_creation
from sklearn.preprocessing import OneHotEncoder
# check version number
from imblearn.over_sampling import RandomOverSampler



def plot_heatmap(df):
    sns.heatmap(df.corr(),annot = True,cmap="PiYG")   # ,cmap= 'plasma'
    # displaying the plotted heatmap
    plt.show()


def z_score_standardization(series):
    return (series - series.mean()) / series.std()

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

def normalize(df):
    
    print ("Normalizing data with min_max_scaling...")
    columns=['Speed','Feature1','Feature2','Feature3','Feature4','Feature5','Feature6']
    # holds the min max values of each column to later be used for normalizg test input
    min_max_values ={} 
    for col in df.columns:
        if col in columns:
            # print(col," in columns")
            min_max_values[col]=[df[col].min(),df[col].max()]     # (min,max) 
            df[col] = min_max_scaling(df[col])
    return df,min_max_values
def prepare_data( remove_description= False,remove_sensor=False,sample='Oversampling'):
    
    df_input = database_creation(lifetime=278) # You can change lifetime here
    #print(df_input.info())
    # print(data['tweet_binary'])
    # print(data['tweet_lifetime'])

    # Remove Event ID nad TimeStamp
    df_input = df_input.drop('TimeStamp', axis=1) 
    df_input = df_input.drop('EventID', axis=1) 
    # SensorsID = to_categorical(df_input['SensorsID'])
    # df_input = df_input.drop('SensorsID', axis=1)
    # df_input['SensorsID']=SensorsID
    # tweet_binary_temp = to_categorical(df_input['tweet_binary'])
    # df_input = df_input.drop('tweet_binary', axis=1)
    # df_input['tweet_binary']=tweet_binary_temp

    if (remove_sensor):
        df_input = df_input.drop('SensorsID', axis=1) 
    
    df_input = df_input.drop('Orientation', axis=1) 
    df_input = df_input.drop('Reason', axis=1) 
    df_input = df_input.drop('Junction', axis=1) 
    
    if (remove_description):
        print("Removing Tweet Data ...")
        df_input = df_input.drop('tweet_binary', axis=1) 
        df_input = df_input.drop('tweet_lifetime', axis=1) 

    # normalize value to help converge faster during training
    df_input , min_max = normalize(df_input)
    df_output = df_input['Class']
    df_input=df_input.drop('Class', axis=1) 

    print("----------------",df_input.info())
    train_x, test_x, train_y, test_y = train_test_split(df_input,df_output,train_size = 0.8,shuffle=True)
    if(sample=='Oversampling'):
       oversample = RandomOverSampler(sampling_strategy=1)
       train_x, train_y = oversample.fit_resample(train_x, train_y)
       #print('Data after oversample:\n ',train_y.value_counts())
       #test_x, test_y = oversamsple.fit_resample(test_x, test_y)
    # elif (sample=='Undersample',train_x.unique()):
    #    undersample=RandomUndersampler(sampling_strategy='minority')
    #    train_x, train_y = undersample.fit_resample(train_x, train_y)
    #    print('Data after undersample: ',train_y['class'].unique())
 
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy()
    test_y = test_y.to_numpy()
    return (train_x, train_y, test_x, test_y,min_max)

if __name__ == "__main__":
       train_x,train_y,test_x,test_y,min_max = prepare_data(False,'Oversampling')



