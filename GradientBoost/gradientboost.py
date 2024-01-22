from DecisionTree import DecisionTree
import pandas as pd
import numpy as np

class GradientBoost:

    def __init__(self,num_models=None,learning_rate=None):
        self.num_models = num_models
        self.alpha = learning_rate


    def base_model(self,X,y):
        
        X.iloc[:,'base_weight'] = y.sum()/len(y)

    def fit(self,X,y):
        self.base_model(X,y)
        X.iloc['r1'] = y-X['base_weight']
        for i in range(self.num_models):
            tree = DecisionTree()
            tree.fit(X.loc[:,:-1],X.loc[:,-1])
            X.loc[:,X.shape[0]] = tree.predict       

    def predict(X):
        return None
    