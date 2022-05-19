from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


# K-fold cross validation 
def KFold_validation(model, X, y, n_splits=5):
    estimator = KerasClassifier(build_fn=model, epochs=200, batch_size=16, verbose=0)
    kfold = KFold(n_splits, shuffle=True)
    results = cross_val_score(estimator, X, y, cv=kfold, verbose=1)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# Definnning classification threshold
def probThreshold(inpProb):
    if inpProb > 0.5:
        return(1)
    else:
        return(0)

def z_score(data):
    # Z-score normalization
    PredictorScaler = StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(data)

    # Generating the standardized values of X
    X = PredictorScalerFit.transform(data)

    return X, PredictorScalerFit


def plot_training_results(training_history):
    # summarize history for accuracy
    plt.plot(training_history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(training_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper left')
    plt.show()
