import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle



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