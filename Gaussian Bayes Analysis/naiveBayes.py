import pandas as pd
import numpy as np

class NaiveBayes:
    def __init__(self):
        self.proir = {}
        self.lhood = {}

    def calculate_prior(self,y):
        classes = np.unique(y)
        prior=[]
        for label in classes:
            l_count = np.where(y==label,1,0).sum()
            prior.append(l_count/float(len(y)))
        return prior
    
    def calculate_likelihood(self,X,ft,ft_val,y,label):
        X = X[y==label]
        ft = X.iloc[:,ft]
        ft_mean = ft.mean()
        ft_std = ft.std()
        #likelihood = (np.exp(np.square((ft_val-ft_mean))/ft_var))/np.sqrt((2*np.pi)*ft_var) 
        #numerator = np.exp(np.square(ft_val-ft_mean)/ft_var)
        #denominator = np.sqrt(2*np.pi*ft_var)
        #likelihood = numerator/denominator
        return (1/(np.sqrt(2*np.pi)*ft_std))*np.exp(-((ft_val-ft_mean)**2/(2*ft_std**2)))
        #return likelihood
    
    # def fit_gaussian(self,X,y):
    #     self.prior = self.calculate_prior(y)
    #     n_feat = X.shape[1]
    #     for label in np.unique(y):
    #         label_likelyhood = {}
    #         for ft in range(n_feat):
    #             ft_likelihood = dict()
    #             unique_attr = np.unique(X[ft])
    #             for val in unique_attr:
    #                 ft_likelihood[val] = self.calculate_likelihood(X,ft,val,y,label)
    #             label_likelyhood[ft] = ft_likelihood
    #         self.lhood[label] = label_likelyhood
    
    # def predict(self,X):
    #     result = []
    #     for j in range(len(X.iloc[:,0])):
    #         x =X.iloc[j,:]
    #         max_prob = 0
    #         ans = 0 
    #         for label in self.labels:
    #             product = self.prior[label]
    #             for i in range(len(x)):
    #                 product *= self.lhood[label][i][]
    #             if max_prob<product:
    #                 max_prob = product
    #                 ans = label
    #         result.append(label)
    #     return result
    
    def fit_gaussian(self,X,y,X_test):
        n_feat = len(X[0])
        labels = np.unique(y)
        ans = []
        priors = self.calculate_prior(y)
        for j in range(len(X_test.iloc[:,0])):
            x = X.iloc[j,:]
            post_prob = np.ones(len(labels))
            for k in  range(len(labels)):
                post_prob[k]*=priors[k]
                for l in  range(len(x)):
                    post_prob[k]*=self.calculate_likelihood(X,l,x[l],y,labels[k])
            ans.append(np.argmax(post_prob))
        return np.array(ans)
        



    