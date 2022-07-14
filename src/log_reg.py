import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pickle

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

class log_model:
    """
    Logistic Regression Model Class that fits a model based on trained data (X_train, y_train) and predict using X_test

    Args:
        X_train [dataframe]: dataframe of training dataset with predictor variables
        y_train [dataframe]: dataframe of training dataset with target variable
        X_test [dataframe]: dataframe of test dataset with predictor variables
        C [float]: regularisation parameter
    
    """
    def __init__ (self,X_train,y_train, X_test,C=1):
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        # initialise log regression
        self.log_reg = LogisticRegression(max_iter=400,C=C)

    def fit (self):
        self.model = self.log_reg.fit(self.X,self.y)

    def predict (self):
        y_pred = self.log_reg.predict(self.X_test);
        return y_pred

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
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    # plt.show()

    return fig

def plot_roc_curve (X_test, y_test, logreg):
    """
    Function to plot ROC Curve

    Args:
        X_test (Series): series object of predictor variables, partitioned into test set
        y_test [Series]: series object for target variables, partitioned into test set

    Returns:
        fig [matplotlib figure]: matplotlib figure object rendered
    """
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba,pos_label='Yes')
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr,tpr,label="auc="+str(auc))
    plt.title('ROC Curve', y=1.1)
    plt.legend(loc=4)
    # plt.show()

    return fig

def plot_feat_impt (logreg, X_encoded):
    """
    Function to plot feature importance

    Args:
        logreg [model instance]: Log Regression model instance
        X_encoded [dataframe]: data with predictor variables encoded from OneHotEncoding()

    Returns:
        fig [matplotlib figure]: matplotlib figure 
    """
    feat_importance = logreg.coef_.flatten()
    fig, ax = plt.subplots(figsize=(10,10))
    # plt.rcParams["figure.figsize"] = (10,10)
    ax.barh(X_encoded.columns, feat_importance, color='g')
    plt.title("Barplot Summary of Feature Importance")
    plt.xlabel("Score")
    # plt.show()
    return fig
    

def log_reg():

    ##### ------  load preprocessed data ------ #####
    df = pd.read_parquet('data/preprocessed.parquet')

    print(f"Load preprocessed data - SUCCESS")

    # split predictor and target variables to X and y respectively
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    ##### ------  Perform Encoding of Categorical Variables ------ #####
    X_encoded = OneHotEncoding(X)
    print(f"Encoding Categorical Variables - SUCCESS")

    ##### ------  Split train and test set ------ #####
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

    ##### ------  Resolve Class Imbalance ------ #####
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    print(f"Balance Class - SUCCESS")

    ##### ------ Fit Logistic Regression model ------ #####
    model_instance = log_model(X_train_smote,y_train_smote,X_test,C=5)
    model_instance.fit() # train model
    y_pred = model_instance.predict() # predict against test set
    print(f"Fit Logistic Regression - SUCCESS")

    ##### ------ Model Evaluation ------ #####
    # Confusion Matrix
    cnf_matrix = plot_cnf_matrix(y_test,y_pred)
    print(f"""
    Accuracy: {metrics.accuracy_score(y_test, y_pred)}
    Precision: {metrics.precision_score(y_test, y_pred,pos_label='Yes')}
    Recall: {metrics.recall_score(y_test, y_pred,pos_label='Yes')}
    Score: {model_instance.model.score(X_test,y_test)}
    """)
    # ROC Curve
    roc_curve = plot_roc_curve(X_test, y_test,model_instance.model)
    # Feature Importance
    feat_impt = plot_feat_impt(model_instance.model,X_encoded)

    ##### ------ Persist Model ------ #####
    # create an iterator object with write permission
    with open('data/model.pkl', 'wb') as files:
        pickle.dump(model_instance, files, pickle.HIGHEST_PROTOCOL)
    
    # save figures
    cnf_matrix.savefig('assets/cnf_matrix.png',bbox_inches='tight')
    roc_curve.savefig('assets/roc_curve.png',bbox_inches='tight')
    feat_impt.savefig('assets/feat_impt.png',bbox_inches='tight')

if __name__ == '__main__':
    log_reg()