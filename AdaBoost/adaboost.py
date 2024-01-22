from decisionTree import DecisionTree
from statistics import mode
import pandas as pd
import numpy as np

class Adaboost:

    def __init__(self,num_stumps=10):
        self.num_stumps = 10
        self.max_depth = 1
        self.stumps = []
    
    def fit(self,X,y):
        
        # breaking condition for the adaboost
        if len(self.stumps)>self.num_stumps:
            return 
        #create the weight for the dataset if the weight not present
        if self.stumps==0 :
            X["sample_weight"] = 1/X.shape[0] 
        # create a stump 
        node = DecisionTree(max_depth=1,n_features=X.shape[1])
        self.stumps.append(node)
        y_predict = node.predict(X)
        incrr_idx = self.incorrect_data(y,y_predict)
        # calculate the performance measure and total error
        preformance_msr = self.performance_measure(X,y,y_predict)
        # update the sample weight
        X['sample_weight'] = self.update_weights(X,preformance_msr)
        #create new dataset for the recursion call 
        self.fit(self.split_data(X,y))
    
    def cummulative_step(self,X_col) :
        return np.cumsum(X_col)

    def split_data(self,X,y):
        X_new= pd.DataFrame()
        y_new = pd.Series()
        cummulative = self.cummulative_step(X['sample_weight'])
        for i in np.random.random_sample(size=X.shape[0]):
            index = np.sort(np.argwhere(cummulative<=i).flatten())[-1]
            X_new.add(X.iloc[index,:])
            y_new.add(y[index])
        return X_new,y_new



    def incorrect_data(self,y,y_predict):
        return np.where(y!=y_predict).flatten()
    
    def cal_total_error(self,X,index):
        total_error = np.sum(X.iloc[index,"sample_weight"])
        return total_error
    
    def performance_measure(self,X,index):
        total_error = self.cal_total_error(X,index)
        return 0.5 * np.log((1-total_error)/total_error)
    
    def update_weights(self,wt_col,per_msr,incr_idx):
        wt_col /= np.exp(per_msr)
        wt_col[incr_idx] *= np.exp(per_msr)*np.exp(per_msr)
        norm_sum =wt_col.sum()
        wt_col /= norm_sum
        return wt_col 
     

    def predict(self,X):
        y_predict = np.array() 
        for i in range(X.shape[0]):
            result = []
            for node in self.stumps:
                result.append(node.predict(X.iloc[i,:]))
            y_predict.add(mode(result))
        return y_predict
            
        
