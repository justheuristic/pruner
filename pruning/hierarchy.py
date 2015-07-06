# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 02:33:10 2015

@author: ayanami
"""
import numpy as np
import scipy as sp
from greedy import *

from sklearn.externals import joblib




from loss_functions import entropy

def select_criteria(factory, thresholds_active,depth,verbose = True):
    """
    I find the most _even_ dichotomies of a given depth among the given thresholds... greedily
    """
    leaves = {"":factory}
    criteria = []

    for i in range(depth):
        entropies = []
        for feature,cut,_ in thresholds_active:
            split_distribution = []
            for leaf in leaves:
                subFactory = leaves[leaf]
                predicate = (subFactory.events[:,feature] > cut)
                trues = sum(subFactory.weights[predicate])
                
                split_distribution.append(trues)
                split_distribution.append(sum(subFactory.weights) - trues)
                
            split_distribution = np.array(split_distribution)/sum(factory.weights)
            entropies.append( entropy(split_distribution))

        feature,cut,_ = criterion = thresholds_active[entropies.index(max(entropies))]
        new_leaves = {}
        
        for leaf in leaves:
            subFactory = leaves[leaf]
            predicate = (subFactory.events[:,feature] > cut)
            new_leaves[leaf+"1"],new_leaves[leaf+"0"] = subFactory.split_by(predicate)

        leaves = new_leaves    
        criteria.append(criterion)
        if verbose:
            print 's:',[leaves[i].events.shape[0] for i in leaves]
            print 'w:',[sum(leaves[i].weights) for i in leaves]
    return criteria
    
def split_upper(factory,criteria,returnIndices = False,equalizeWeights = False,normalizeWeights = False,
                       split_weights = 0.,split_inclusion = 0.):
    """
    I split the data into the leaves of the upper level ODT given it's criterea
    """
    split = {'':factory}
    indices = {'':np.arange(factory.n_events)}
    for criterion in criteria:
        feature,cut,_= criterion
        split_new = {}
        indices_new = {}
        for splt in split:
            dichotomy = split[splt].features[:,feature][:split[splt].n_events]> cut
            split_new[splt+'1'],split_new[splt+'0'] = split[splt].split_by(dichotomy,split_weights,split_inclusion)
            if returnIndices:
                indices_new[splt+'1'] = indices[splt][dichotomy]
                indices_new[splt+'0'] = indices[splt][dichotomy == False]
        split = split_new
        indices = indices_new
    if equalizeWeights:
        for splt in split:
            split[splt].equalizeWeights()
    if normalizeWeights:
        for splt in split:
            split[splt].normalizeWeights()
        
    return ((split, indices ) if returnIndices else split)

def train_splitted_boosts( trees,
                           factory,
                           criteria,
                           loss,
                           learning_rate=0.25,
                           breadth=0.25,
                           nTrees_leaf=100,
                           trees_sample_size=100,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.,
                           weights_outside_leaf = 0.,
                           inclusion_outside_leaf = 0.,
                           use_joblib = False,
                           joblib_method = "threads",
                           use_joblib_inner = False,
                           joblib_method_inner = "threads",
                           copy_pred_inner = False,
                           n_jobs = -1,
                           n_jobs_inner=-1,
                           initialTrees= None):
    """
    make greedy prune for every leaf in split. I know i should be with kwargs
    """

    factories = split_upper(factory,criteria,
                                   split_weights = weights_outside_leaf,
                                   split_inclusion= inclusion_outside_leaf)
    
    leaves = factories.keys()
    if initialTrees == None:
        initialTrees = {leaf:[] for leaf in leaves}
    
        
    if not use_joblib:
        classis = []
        itr = 1
        for leaf in leaves:
            if verbose:
                print "\n\nNow training leaf ",leaf, itr,"/",len(leaves)
                print "n_ samples at leaf = ", factories[leaf].n_events
                itr +=1
            classi = greed_up_features_bfs(
                               trees,
                               factories[leaf],
                               loss,
                               learning_rate,
                               breadth,
                               nTrees_leaf,
                               trees_sample_size,
                               verbose,
                               learning_rate_decay,
                               trees_sample_increase,
                               regularizer = regularizer,
                               use_joblib = use_joblib_inner,
                               n_jobs = n_jobs_inner,
                               joblib_method = joblib_method_inner,
                               copy_pred = copy_pred_inner,
                               initialBunch = initialTrees[leaf]
                               )

            classis.append(classi)
    else: #use joblib
        if joblib_method not in ["threads","processes"]:
            raise ValueError, "please specify a valid backend type: threads or processes"
        tasks = [joblib.delayed(greed_up_features_bfs)(
                               copy.deepcopy(trees),
                               factories[leaf],
                               copy.deepcopy(loss),
                               learning_rate,
                               breadth,
                               nTrees_leaf,
                               trees_sample_size,
                               False,
                               learning_rate_decay,
                               trees_sample_increase,
                               regularizer = regularizer,
                               use_joblib = use_joblib_inner,
                               n_jobs = n_jobs_inner,
                               joblib_method = joblib_method_inner,
                               copy_pred = copy_pred_inner,
                               initialBunch = copy.deepcopy(initialTrees[leaf])#just in case it's the same...
                               )for leaf in leaves]
                               
        
        classis = joblib.Parallel(n_jobs = n_jobs,
                                  backend = "threading" if joblib_method == "threads" else "multiprocessing",
                                  verbose=verbose)(tasks)


    return {leaves[i]:classis[i] for i in range(len(leaves))}
    
    
def wheel_splitted_boosts( initialTrees,
                           trees,
                           factory,
                           criteria,
                           loss,
                           learning_rate,
                           wheel_up_times,
                           trees_sample_size,
                           random_walk= True,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.,
                           weights_outside_leaf = 0.,
                           inclusion_outside_leaf = 0.,
                           use_joblib = False,
                           use_joblib_inner = False,
                           joblib_method_inner = "threads",
                           copy_pred_inner = False,
                           n_jobs = -1,
                           n_jobs_inner=-1,
                           ):
    """
    make greedy prune for every leaf in split. I know i should be with kwargs
    """

    factories = split_upper(factory,criteria,
                                   split_weights = weights_outside_leaf,
                                   split_inclusion= inclusion_outside_leaf)
    
    leaves = factories.keys()

    
        
    if not use_joblib:
        classis = []
        itr = 1
        for leaf in leaves:
            if verbose:
                print "\n\nNow wheeling leaf ",leaf, itr,"/",len(leaves)
                print "n_ samples at leaf = ", factories[leaf].n_events
                itr +=1
            classi = wheel_up_features_bfs(
                               initialTrees[leaf],
                               trees,
                               factories[leaf],
                               loss,
                               learning_rate,
                               wheel_up_times,
                               trees_sample_size,
                               verbose = verbose,
                               learning_rate_decay = learning_rate_decay,
                               trees_sample_increase = trees_sample_increase,
                               regularizer = regularizer,
                               random_walk = random_walk,
                               use_joblib = use_joblib_inner,
                               n_jobs = n_jobs_inner,
                               joblib_method = joblib_method_inner,
                               copy_pred = copy_pred_inner)

            classis.append(classi)
    else: #use joblib
        tasks = [joblib.delayed(wheel_up_features_bfs)(
                               initialTrees[leaf],
                               trees,
                               factories[leaf],
                               loss,
                               learning_rate,
                               wheel_up_times,
                               trees_sample_size,
                               verbose = False,
                               learning_rate_decay = learning_rate_decay,
                               trees_sample_increase = trees_sample_increase,
                               regularizer = regularizer,
                               random_walk = random_walk,
                               use_joblib = use_joblib_inner,
                               n_jobs = n_jobs_inner,
                               joblib_method = joblib_method_inner,
                               copy_pred = copy_pred_inner)for leaf in leaves]
        classis = joblib.Parallel(n_jobs = n_jobs,
                                  backend = "threading",
                                  verbose=verbose)(tasks)


    return {leaves[i]:classis[i] for i in range(len(leaves))}
        


def predict_splitted(factory,criteria,trees):
    """predict with a splitted trees boost"""
    factories, indices =  split_upper(factory,criteria,True)
    y_pred = np.zeros(factory.n_events)
    for leaf in factories.keys():
        y_pred[indices[leaf]] = factories[leaf].predict(trees[leaf])
    return y_pred
    
