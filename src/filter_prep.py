from random import shuffle
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


def unique_values(df):
    for column in df.columns:
        df_col = df[column]
        uniqe_values = df_col.unique()
        #print ("column "+ str(column) + " unique values are" + str (uniqe_values))
    

def split_reason(row):
    # if we split the decription based on ":"  3 columns will be created since we have 2 ":"
    split = row['description'].split(':')
    return split[2]

# def plot_heatmap(df):
#     sns.heatmap(df.corr(),annot = True,cmap="PiYG")   # ,cmap= 'plasma'
#     # displaying the plotted heatmap
#     plt.show()


def z_score_standardization(series):
    return (series - series.mean()) / series.std()

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

def normalize(df):
    
    #print ("Normalizing data with min_max_scaling...")
    columns=['Speed','Feature1','Feature2','Feature3','Feature4','Feature5','Feature6']
    # holds the min max values of each column to later be used for normalizg test input
    min_max_values ={} 
    for col in df.columns:
        if col in columns:
            # print(col," in columns")
            min_max_values[col]=[df[col].min(),df[col].max()]     # (min,max) 
            df[col] = min_max_scaling(df[col])
    return df,min_max_values
def prepare_data( file, remove_descriptions=True,sample='none'):
    # import csv file
    df =  pd.read_csv(open(file),index_col=False)
    # print(df.info(),"Shape_of_dataframe : " + str (df.shape),df.dtypes,sep='\n ////////////////////////////////// \n')

    # check for rows that have null values 
    df.dropna(inplace=True)
    # change the datatype of class,EventID,SensorsID from float to integer
    df['EventID'] = df['EventID'].astype('int64')
    df['SensorsID'] = df['SensorsID'].astype('int64')
    df['Class'] = df['Class'].astype('int64')
    df['TimeStamp'] = df['TimeStamp'].astype('int64')
    # divide description into junction,orientation, and reason 

    # -> check if the format of the description is the same by comparing the number of unique values
    split_description = []
    reason = []
    for item in df['Description']:
        split = item.split()
        # if we split the decription based on ":"  3 columns will be created since we have 2 ":"  3
        # This helps keep reason words together, after 2nd ":" the reason is mentioned
        split_colon = item.split(':')
        reason.append(split_colon[2])# reason is after the secocnd colon
        split_description.append(split)
    df_split = pd.DataFrame(split_description)
    # print (df_split.info())

    # simple spliting shows that the decription has different size of words for differnt events
    # lets check the number of unique values in each column
    unique_values(df_split)
    df_split.drop(labels=[0,1,2,3], axis=1, inplace=True)
    #print(df_split.info()) #check the number of columns before droping

    # add column 4 (orientation) as a column in the main data


    df ['Orientation'] = df_split[4] # add orientation as a feature

    # remove column orientation from the main data
    df_split.drop(labels=[4], axis=1, inplace=True)
 

    df['Junction'] = df_split.apply(lambda row: row[7] if row[5]=="at" else row[7]+","+row[9], axis=1) 
    df['Reason'] = reason
    #print(df_split.info())
    # print(df.loc[29794])     # a good trial sample
    ''' Remove Description from data since we have all the necesarry data extracted
    The following three features were extracted -> orientation,junction and reason'''

    df.drop(columns=['Description'], inplace=True)

    ''' Now we have 14 features 6 features named Feature1 - Feature6 in addition to 
    EventID(int),SensorID(int),TimeStamp(int),Class(0|1),Speed(float),Orientation(obj),Junction(obj),Reason(obj)'''
    # First lets change the datatype orientation,junction and reason into categorical datatypes
    # This means the values of those features will be hot encoded 
    # Notice the memory size decrease after using catgorical (2.9MB) data instead of object or string (3.6+MB) 
    df['Orientation'] = df['Orientation'].astype("category")
    df['Junction'] = df['Junction'].astype("category")
    df['Reason'] = df['Reason'].astype("category")
    df['SensorsID'] = df['SensorsID'].astype("category")



    words_list =[]
    words_dic ={}
    start_time,end_time = {},{} 
    event_start,event_end = {},{}
    start_end, event_time = {},{}
  
    

    for index, row in df.iterrows():
        if row['EventID'] in start_time:
            event_id = row['EventID']
            end_time[event_id] = row['TimeStamp']
        else: 
            # timestamp = row['TimeStamp']
            # dt_object = datetime.fromtimestamp(timestamp)
            event_id = row['EventID']
            start_time[event_id] = row['TimeStamp']
            temp_word = []
            for junc in row['Junction'].split(','):
                temp_word.append(junc)
            temp_word.append(row['Reason'])
            temp_word.append(row['Orientation'])
            words_list.append(temp_word)
            words_dic[event_id] = temp_word
            
            # Check if event is not in event start array and class is 1
            # if event is 1 and not in the array this is the beginning of class 1
        if (row['Class'] == 1 and row['EventID'] not in event_start):
            event_id = row['EventID']
            event_start[event_id] = row['TimeStamp']
        elif(row['Class'] == 1):
            event_id = row['EventID']
            event_end[event_id] = row['TimeStamp']

    for k in event_start.keys():
        # Time an event (class 1) started and ended
        event_time[k] = (event_start[k], event_end[k])
    for k in start_time.keys():
         # Time an event (classes 0-1-0) started and ended (before and after event)
        start_end[k] = (start_time[k], end_time[k])

    #unique_values(df)
    # print(df.head(5))
    # print("----------------------------------------------------------------")
    # print(words_dic)
    # print("----------------------------------------------------------------")
    # print(start_end)
    # print("-----------------------------------------------------------------")
    # print(event_time)
    return words_dic,start_end,event_time


# prepare_data()