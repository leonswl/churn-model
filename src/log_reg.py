from sklearn.linear_model import LogisticRegression


class log_model:
    def __init__ (self,X_train,y_train, X_test):
        self.X = X_train
        self.y = y_train
        self.X_text = X_test
        # initialise log regression
        self.log_reg = LogisticRegression()

    def fit (self):
        self.model = self.log_reg.fit(self.X,self.y)

    def predict (self):
        y_pred = self.log_reg.predict(self.X_test);
        return y_pred


model_instance = log_model(X_train_smote,y_train_smote)
model_instance.fit()
print(model_instance.predict())