import pandas as pd
import numpy as np

class NaiveBayes:

    def __init__(self):
        self.dictionary = {}
        self.likelihood = {}
        self.prior = {}
        self.labels = []
        self.class_count = {}
    
    def calculate_prior(self,y):
        denominator = len(y)+len(self.labels)
        for label in self.labels:
            freq = 1+(np.where(y==label,1,0).sum())
            self.class_count[label] = freq
            self.prior[label] = freq/denominator

    def create_prior_dicts(self,y):
        self.labels = list(np.unique(y))
        for label in self.labels:
            self.dictionary[label] = {}
            self.likelihood[label] = {}

    def fit(self,X,y):
        # create dictionaries
        self.create_prior_dicts(y)
        # enumerate the data and create the data
        for i in (X.index):
           data = self.get_words(X.loc[i,'text'])
           self.update_dictionary(data=data,label = y[i])
        # get prior probabilities
        self.calculate_prior(y=y)
        # get likelihood probabilities
        for label in self.labels:
            base_y = np.where(y==label,1,0).sum()
            for word in self.dictionary[label].keys():
                self.likelihood[label][word] = (self.dictionary[label][word]+1) / (base_y+1)     

    def get_words(self,x):
        return x.split(' ')
    
    def update_dictionary(self,data,label):
        for word in data:
            if word in self.dictionary.keys():
                self.dictionary[label][word] +=1
            else :
                self.dictionary[label][word] = 1
    
    def predict(self,X):
        results = []
        for j in X.index:
            data = self.get_words(X.loc[j,'text'])
            post_prob = np.ones(len(self.labels))
            max_prob = -100000
            ans = None
            for i in range(len(self.labels)):
                post_prob[i] *= self.prior[self.labels[i]]
                for word in data:
                    if word in self.likelihood[self.labels[i]].keys():
                        post_prob[i]*=self.likelihood[self.labels[i]][word]
                    else:
                        post_prob[i]*= 1/self.class_count[self.labels[i]]
                if max_prob<post_prob[i]:
                    max_prob = post_prob[i]
                    ans = self.labels[i]
            results.append(ans)
        return np.array(results)

   
    
    
    