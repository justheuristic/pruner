# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 02:19:41 2015

@author: ayanami
"""
import numpy as np
import scipy as sp

class DataFactory:
    def __init__(self,events,labels,weights = None):
        self.events = events
        self.labels = labels

        if weights is None:
            weights = np.ones(labels.shape)
        self.weights = weights
        
        # extending the data so the number of events is divisible by 8
        self.n_events = len(events)
        self.n_extended64 = (self.n_events + 7) // 8
        self.n_extended = self.n_extended64 * 8

        # using Fortran order (surprisingly doesn't seem to influence speed much)
        self.features = np.zeros([self.n_extended, self.events.shape[1]], dtype='float32', order='F')
        self.features[:self.n_events, :] = self.events
    def predict(self,formula):
        '''
        make real-value predictions using sklearn gradient_boosting.predict_stages
        '''
        if len(formula) ==0:
            return np.zeros(self.events.shape[0])+formula.bias
        result = np.zeros(len(self.events), dtype=float)
        for stage_predictions in self.apply_separately(formula):
            result += stage_predictions
        return result
    def apply_separately(self,formula):
        """
        :param events: numpy.array (or DataFrame) of shape [n_samples, n_features]
        :return: each time yields numpy.array predictions of shape [n_samples]
            which is output of a particular tree
        """
        # result of first iteration
        yield np.zeros(len(self.events), dtype=float)+ formula.bias
        
        nf_count = len(formula)
        if nf_count ==0:return
        
        tree_depth = len(formula[0][0])
        
        
        for tree in formula:
            leaf_indices = self.getLeafIndices(tree)        
            leaf_values = tree[2]
            yield leaf_values[leaf_indices]
            
    def getLeafIndices(self,tree):
        """
        get the tree active leaf indices on the data
        """
        tree_features, tree_cuts, _ = tree
        leaf_indices = np.zeros(self.n_extended64, dtype='int64')
            
        for tree_level, (feature, cut) in enumerate(zip(tree_features, tree_cuts)):
            leaf_indices |= (self.features[:, feature] > cut).view('int64') << tree_level
        return leaf_indices.view('int8')[:self.n_events]
    def equalizeWeights(self):
        """modify weights so that positive and negative class are weighted equally and all weights sum to 1"""
        sum_p = sum(self.weights[self.labels==1])
        sum_n = sum(self.weights)-sum_p
        self.weights[self.labels == 1] *= (sum_n/sum_p)/(sum_n*2)
        self.weights[self.labels == 0] *= 1./(sum_n*2)
    def normalizeWeights(self):
        """forces sum of weights to 1 without changing proportions"""
        self.weights/=sum(self.weights)
    def split_by(self,boolean,offclass_weights=0.,offclass_sample=0.,indices =None):
        """splits dataset by a boolean array in an unstrict manner:
            offclass_weights: a multiplier for weights of samples from foreign part
            offclass_sample: what fraction of foreign samles are to be included in each part (probability [0;1])
        """
        inverse = boolean == False
        
        #what indices to include as offclass ones
        num_p = sum(boolean)
        num_n = sum(inverse)
        assert num_p !=0 and num_n !=0

        boolean_addition_indices = np.random.choice(range(num_p),int(num_p*offclass_sample),replace=False)
        inverse_addition_indices = np.random.choice(range(num_n),int(num_n*offclass_sample),replace=False)
        
        FactoryClass = self.__class__
                                                    
        factory_p = FactoryClass(np.append(self.events[boolean],self.events[inverse][inverse_addition_indices],0),
                                np.append(self.labels[boolean],self.labels[inverse][inverse_addition_indices],0),
                                np.append(self.weights[boolean],self.weights[inverse][inverse_addition_indices]*offclass_weights,0))
        
        factory_n = FactoryClass(np.append(self.events[inverse],self.events[boolean][boolean_addition_indices],0),
                                np.append(self.labels[inverse],self.labels[boolean][boolean_addition_indices],0),
                                np.append(self.weights[inverse],self.weights[boolean][boolean_addition_indices]*offclass_weights,0))
        
        if indices is not None:
            indices_p = np.append(indices[boolean],indices[inverse][inverse_addition_indices],0)
            indices_n = np.append(indices[inverse],indices[boolean][boolean_addition_indices],0)
            return ((factory_n,indices_n),(factory_p,indices_p))
                                  
        #it is CRITICAL that first comes the b==0 child and than b==1 one. 
        #Don't change that unless you are sure about it.
        return (factory_n,factory_p) 
        
    def select_by(self,selector):
        return self.__class__(self.events[selector],self.labels[selector],self.weights[selector])
    
class BinaryClassificationFactory(DataFactory):
    def __init__(self,events,labels,weights):
        DataFactory.__init__(events,labels,weights)
        self.labels_sign = labels*2 -1
    
class RegressionFactory(DataFactory):
    pass
