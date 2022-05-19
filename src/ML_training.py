from training_data_prep_ml import prepare_data
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from filter_tweets import database_creation
# from xgboost import XGBClassifier
import xgboost as xgb
import seaborn as sns
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
def print_results(trained_model,input,output,predicted,train=True):
    
    if train:
        print("--------------------- Training------------------------ ")
    else:
        print("--------------------- Testing------------------------ ")

    print("Model accuracy:{0:.2f}% ".format(trained_model.score(input, output)*100))
    print("Model precision:{0:.2f}% ".format(precision_score(output,predicted,average='binary')*100))
    print("Model Recall:{0:.2f}% ".format(recall_score(output,predicted)*100))
    print("Model F1_score:{0:.2f}% ".format(f1_score(output,predicted)*100))

def train_logistic_reg(remove_description,remove_sensor,sample='Oversampling'):
    logistic_model = LogisticRegression(random_state=10,max_iter=1000) #max_iter=10000
    print("Preparing Training Data ...")
    train_x,train_y,test_x,test_y,min_max = prepare_data(remove_description,remove_sensor,sample)
    print("Trainning ...")
    logistic_model.fit(train_x, train_y.ravel())
    

    logistic_test_pred=logistic_model.predict(test_x)
    logistic_train_pred=logistic_model.predict(train_x)
    cf_matrix=confusion_matrix( test_y,logistic_test_pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')
    #plt.show()
    # print result for training and testing
    #print_results(logistic_model,train_x,train_y,logistic_train_pred,True)
    print_results(logistic_model,test_x,test_y,logistic_test_pred,False)
def train_knn(remove_description,remove_sensor,sample='Oversampling'):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    print("Preparing Training Data ...")
    train_x,train_y,test_x,test_y,min_max = prepare_data(remove_description,remove_sensor,sample)
    print("Trainning ...")
    knn_model.fit(train_x, train_y.ravel())

    knn_test_pred=knn_model.predict(test_x)
    knn_train_pred=knn_model.predict(train_x)
    cf_matrix=confusion_matrix( test_y,knn_test_pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    #print_results(knn_model,train_x,train_y,knn_train_pred,True)
    print_results(knn_model,test_x,test_y,knn_test_pred,False)
def train_decisionTree(remove_description,remove_sensor,sample='Oversampling'):
    tree_model = DecisionTreeClassifier()
    print("Preparing Training Data ...")
    train_x,train_y,test_x,test_y,min_max = prepare_data(remove_description,remove_sensor,sample)
    print("Trainning ...")
    tree_model.fit(train_x, train_y.ravel())

    tree_test_pred=tree_model.predict(test_x)
    tree_train_pred=tree_model.predict(train_x)
    cf_matrix=confusion_matrix( test_y,tree_test_pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    #print_results(tree_model,train_x,train_y,tree_train_pred,True)
    print_results(tree_model,test_x,test_y,tree_test_pred,False)
def train_RandomForest(remove_description,remove_sensor,sample='Oversampling'):
    forest_model = RandomForestClassifier()
    print("Preparing Training Data ...")
    train_x,train_y,test_x,test_y,min_max = prepare_data(remove_description,remove_sensor,sample)
    print("Trainning ...")
    forest_model.fit(train_x, train_y.ravel())

    forest_test_pred=forest_model.predict(test_x)
    forest_train_pred=forest_model.predict(train_x)
    cf_matrix=confusion_matrix( test_y,forest_test_pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    #print_results(forest_model,train_x,train_y,forest_train_pred,True)
    print_results(forest_model,test_x,test_y,forest_test_pred,False)
def train_GBS(remove_description,remove_sensor,sample='Oversampling'):
    gradient_model = GradientBoostingClassifier()
    print("Preparing Training Data ...")
    train_x,train_y,test_x,test_y,min_max = prepare_data(remove_description,remove_sensor,sample)
    print("Trainning ...")
    gradient_model.fit(train_x, train_y.ravel())

    gradient_test_pred=gradient_model.predict(test_x)
    gradient_train_pred=gradient_model.predict(train_x)
    cf_matrix=confusion_matrix( test_y,gradient_test_pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    #print_results(gradient_model,train_x,train_y,gradient_train_pred,True)
    print_results(gradient_model,test_x,test_y,gradient_test_pred,False)
def train_XGB(remove_description,remove_sensor,sample='Oversampling'):
    xgb_model = xgb.XGBClassifier(use_label_encoder=False)
    print("Preparing Training Data ...")
    train_x,train_y,test_x,test_y,min_max = prepare_data(remove_description,remove_sensor,sample)
    
    xgb_model.fit(train_x, train_y.ravel())

    xgb_test_pred=xgb_model.predict(test_x)
    xgb_train_pred=xgb_model.predict(train_x)
    cf_matrix=confusion_matrix( test_y,xgb_test_pred)

    labels = ['True Neg','False Pos','False Neg','True Pos']
    sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues')

    #print_results(xgb_model,train_x,train_y,xgb_train_pred,True)
    print_results(xgb_model,test_x,test_y,xgb_test_pred,False)
    
def main():
    remove_description = False
    # Retrive data ( dataframe format) 
    # data = database_creation(lifetime=278) # You can change lifetime here
    # print(data.info())
    # print(data['tweet_binary'])
    # print(data['tweet_lifetime'])
    sample='Oversampling'  #'none' #'Oversampling'
    remove_sensor=False
    i=0
    while(i<2):
        # print("-------------------- Logestic Regression With Tweets-------------------------",not(remove_description))
        # train_logistic_reg(remove_description,remove_sensor,sample)
        print("-------------------- KNN Model With Tweets-------------------------",not(remove_description))
        train_knn(remove_description,remove_sensor,sample)
        print("-------------------- Decision Tree Model With Tweets-------------------------",not(remove_description))
        train_decisionTree(remove_description,remove_sensor,sample)
        print("-------------------- Random Forest Model With Tweets-------------------------",not(remove_description))
        train_RandomForest(remove_description,remove_sensor,sample)
        print("-------------------- Gradient Boosting Model With Tweets-------------------------",not(remove_description))
        train_GBS(remove_description,remove_sensor,sample)
        print("-------------------- XGB Model With Tweets-------------------------",not(remove_description))
        train_XGB(remove_description,remove_sensor,sample)
        remove_description=True
        i+=1


if __name__ == "__main__":
    main()