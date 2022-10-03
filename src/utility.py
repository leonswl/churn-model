# utility scripts
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def OneHotEncoding(data):
    """
    Function to perform One Hot Encoding on Categorical Variables

    Args: 
        data [dataframe]: data with only predictor variables to have categorical attributes encoded

    Returns:
        data_encoded [dataframe]: data with predictor variables encoded
    """
    # create copy of data
    data_encoded = data.copy()

    # grab categorical column names
    cat_vars = data_encoded.select_dtypes(include=['object']).columns.tolist()    

    # encode 
    for var in cat_vars:
        data_encoded = pd.get_dummies(data_encoded,columns=[var])

    return data_encoded


def apply_smote (X,y):
    """
    Function to apply SMOTE on dataset

    Args:
        X [dataframe]: dataframe with predictor variables and can only contain continuous variables. NaN values are not allowed
        Y [dataframe]: dataframe with target variables. NaN values are not allowed

    Returns:
        x_smote_df [dataframe]: final predictor dataset with SMOTE
        y_smote_df [dataframe]: final target dataset with SMOTE
    
    """
    

    try:
        # initialise SMOTENC
        smote_nc = SMOTE()

        # fit predictor and target variable
        x_smote, y_smote = smote_nc.fit_resample(X.to_numpy(),y.to_numpy())

    except Exception as e:
        print(e)

    finally:
        # convert x and y arrays to dataframe
        x_smote_df = pd.DataFrame(x_smote,columns=X.columns)
        y_smote_df = pd.DataFrame(y_smote,columns=[y.name])

        return x_smote_df, y_smote_df


def plot_cnf_matrix (y_test,y_pred):
    """
    Function to plot confusion matrix

    Args:
        y_test [Series]: series object for target variables, partitioned into test set
        y_pred [Array or Series]: array or series object of predicted target variable. Length must be the same as y_test
    
    Returns:
        fig (matplotlib figure): matplotlib figure object rendered
    
    """

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), xticklabels=class_names, yticklabels=class_names, annot=True, cmap="YlGnBu" ,fmt="d")
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    # plt.show()

    return fig