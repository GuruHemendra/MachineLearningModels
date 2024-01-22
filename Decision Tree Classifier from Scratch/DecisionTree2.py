import numpy as np
import pandas as pd

class Node:
    def __init__(self,threshold=None,feature=None,*,value=None):
        self.threshold = threshold
        self.left = None
        self.right = None
        self.value = value
        self.feature = feature
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self,max_depth=100,min_scale=2,n_feature=None):
        self.root = None
        self.max_depth = max_depth
        self.min_scale = min_scale
        self.n_feature = n_feature
    

    def fit(self,X,y):
        #self.n_feature = X.shape[1] if not self.n_feature else min(self.n_feature,X.shape[1])
        self.n_feature = X.shape[1] if not self.n_feature else min(X.shape[1],self.n_feature)
        self.root = self._grow_tree(X,y)


    def _grow_tree(self,X,y,depth =0):
        n_label = len(np.unique(y))
        n_feat = X.shape[1]
        # checking the stopping criteria for the tree
        if self.max_depth<=depth or self.min_scale>=len(y) or n_label==1:
            left_value = self.most_common_value(y)
            return Node(value = left_value) 
        # step 1: getting the best split condition
        feat_idx = np.random.randint(low = 0,high = n_feat,size = self.n_feature)
        best_split_feat,best_split_threshold = self.best_split(X,y,feat_idx)
        # step 2: applying the split condition
        newNode = Node(threshold=best_split_threshold,feature=best_split_feat)
        l_idx,r_idx = self._split_data(X.iloc[:,best_split_feat],best_split_threshold)
        newNode.left = self._grow_tree(X[l_idx,:],y[l_idx],depth+1)
        newNode.right = self._grow_tree(X[r_idx],y[r_idx],depth+1)
        return newNode


    def best_split(self,X,y,feat_idx):
        best_split_feat = -1
        best_gain = 0
        best_gain_threshold = -1
        for feat in feat_idx:
            # calculate the information gain for feature
            X_col = X.iloc[:,feat]
            thresholds = np.unique(X_col) 
            for thr in thresholds:
                # calculate information to this feature i and threshold j 
                gain = self.information_gain(X,y,feat,thr)
                if best_gain<gain:
                    best_gain_threshold = thr
                    best_split_feat = feat
                    best_gain = gain
        return best_split_feat,best_gain_threshold
                

    def _entropy(self,y):
        counts = y.value_counts().to_dict()
        ps = [counts[x]/len(y) for x in counts]
        en = [p*np.log(p) for p in ps]
        return sum(en)
    

    def information_gain(self,X,y,feat,thr):
        # parent_entropy - childentropy*weighted
        parent_entropy = self._entropy(y)
        parent_weight = len(y)
        #split child based on the feature and threshold
        l_idx,r_idx = self._split_data(X[:,feat],thr)
        child_entropy_sum = (len(l_idx)/len(y))*self._entropy(y[l_idx])
        child_entropy_sum += (len(r_idx)/len(y))*self._entropy(y[r_idx]) 
        return parent_weight - child_entropy_sum


    def _split_data(self,X_col,threshold):
        left_idx = np.argwhere(X_col<=threshold).flatten()
        right_idx= np.argwhere(X_col>threshold).flatten()
        return left_idx,right_idx
        

    def most_common_value(self,y):
        labels = pd.DataFrame(y,columns=["y"])
        return labels.y.value_counts().sort_values(ascending=False).index[0]
    
    def predict(self,X):
        return np.array([self.traversetree(self.root,x) for x in X])
    
    def traversetree(self,node,x):
        if node.is_leaf_node():
            return node.value
        if x[node.feature]<=node.threshold:
            return self.traversetree(node.left,x)
        else:
            return self.traversetree(node.right,x)

