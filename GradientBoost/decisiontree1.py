import pandas as pd
import numpy as np
from statistics import mode

class Node:
    
    def __init__(self,feature=None,thr=0,*,value=None):
        self.feature = feature
        self.thr = thr
        self.left = None
        self.right = None
        self.value = value

class DecisionTree1:

    def __init__(self,num_feature=None,max_depth=3,min_scale=5,criteria='entropy'):
        self.num_feature = num_feature
        self.max_depth = max_depth
        self.min_scale = min_scale
        self.root = None
        self.criteria = criteria
        self.select_fun = {'entropy':self.information_gain, 'gini':self.gini_impurity}

    def fit(self,X,y):
        if self.num_feature==None:
            self.num_feature=X.shape[1]
        else:
            self.num_feature = min(X.shape[1],self.num_feature)
        self.root = self.build_tree(X,y)

    def build_tree(self,X,y,depth=0):
        if self.max_depth<=depth or self.min_scale>len(y) or len(np.unique(y))==1 :
            return Node(self.most_common(y))
        feat = np.random.choice(X.shape[1],size=self.num_feature,replace=True)
        max_gain = -1
        best_feat = 0
        best_thrs = 0
        #print("building tree")
        for ft in feat:
            thrs = np.unique(X.iloc[:,ft])
            if len(thrs)>0:
                #print("entered thr")
                for thr in thrs:
                    #print(" in second")
                    gain = self.select_fun[self.criteria](X,y,ft,thr)
                    if gain>max_gain:
                        best_thrs = thr
                        max_gain = gain
                        best_feat = ft
                        #print("changed thr :",best_feat,best_thrs)
        node = Node(feature=best_feat,thr=best_thrs)
        l_idx,r_idx = self.split_data(X.iloc[:,best_feat],best_thrs)
        if len(l_idx)!=0  and len(r_idx)!=0:
            node.left = self.build_tree(X.loc[l_idx,:],y[l_idx],depth+1)
            node.right = self.build_tree(X.loc[r_idx,:],y[r_idx],depth+1)
        return node
    
    def split_data(self,X_col,thr):
        l_idx = np.where(X_col<=thr,1,0)
        r_idx = np.where(X_col>thr,1,0)
        return l_idx,r_idx
    
    def gini_impurity(self,X,y,feat,thrs):
        l_idx,r_idx = self.split_data(X.iloc[:,feat],thrs)
        # print(l_idx.sum())
        # print(r_idx.sum())
        #print(len(y))
        ps = [np.sum(l_idx)/len(y),np.sum(r_idx)/len(y)]
        #print(ps)
        ps2 = [p*p for p in ps]
        #print(1-sum(ps2))
        return 1-sum(ps2)
    
    def entropy(self,y):
        labels = np.unique(y)
        ps = [np.sum(y==x)/len(y) for x in labels]
        return np.sum(p*np.log(p) for p in ps)
    
    def information_gain(self,X,y,feat,thr):
        parent_en = self.entropy(y)
        l_idx,r_idx = self.split_data(X.iloc[:,feat],thr)
        l_wt,r_wt = len(l_idx)/len(y), len(r_idx)/len(y)
        l_en = self.entropy(y[l_idx]) 
        r_en = self.entropy(y[r_idx])
        child_en = l_en*l_wt + r_en*r_wt
        return parent_en - child_en
    
    def most_common(self,y):
        return mode(y)
    
    def predict(self,X):
        return [self.traverse(X.iloc[i,:],self.root) for i in range(X.shape[0])]
    
    def traverse(self,x,node):
        # if node is None or node.thr is None or node.feature is None:
        #     return 0
        if node.value is not None:
            return node.value
        else :
            if x[node.feature]<=node.thr:
                return self.traverse(x,node.left)
            return self.traverse(x,node.right)


    def print_tree(self,node,level):
        if node is not None:
            self.print_tree(node.left,level+1)
            print(level,"\t\t",node.feature,"\t\t",node.thr)
            self.print_tree(node.right,level+1)

    def plot_tree(self):
        self.print_tree(self.root,0)

