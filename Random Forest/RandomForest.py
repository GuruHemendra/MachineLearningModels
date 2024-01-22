from decisiontree import DecisionTree1
from collections import Counter
import numpy as np

class RandomForest:

    def __init__(self,num_trees=5,min_scale=2,max_depth=100,n_features=None):
        self.num_trees = num_trees
        self.min_scale = min_scale
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees_obj = []
    
    def fit(self,X,y):
        #calculate samplesize
        sample_size = X.shape[0]//self.num_trees
        for i in range(self.num_trees):
            tree = DecisionTree1(self.max_depth,self.n_features,self.min_scale)
            #get the data indexs
            rows,cols = self.split_data(X,sample_size)
            X_train = X.loc[rows,cols]
            y_train = y[rows]
            tree.fit(X_train,y_train)
            self.trees_obj.append(tree)

    def predict(self,X):
        return [self.generate(X.iloc[i,:]) for i in range(X.shape[0])]

    def generate(self,x):
        result = []
        for tree in self.trees_obj:
            result.append(tree.predict(x))
        return Counter(result).most_common(1)[0][0]

    def split_data(self,X,size):
        rows = np.random.choice(X.index,size=size)
        cols = np.random.choice(X.columns,size=np.random.randint(low=2,high=X.shape[1]))
        return rows,cols
    

