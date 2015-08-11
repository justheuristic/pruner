# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 02:30:48 2015

@author: ayanami
"""

from sklearn.externals import joblib
import numpy as np
import random,copy
from factory import BinaryClassificationFactory,RegressionFactory




class PrunedFormula(list):
    '''gradient boosting formula with minimal features;
    a list with bias'''
    def __init__(self,trees,bias = 0):
        self.bias = bias
        list.__init__(self,trees)
    def __repr__(self):
        return str(self.bias)+' '+list.__repr__(self)
    def predict(self,factory):
        return factory.predict(self)
    def __add__(self,other):
        assert type(other) is list
        return PrunedFormula(list(self)+other,self.bias)
    def staged_predict(self,factory):
        return factory.apply_separately(self)
    


def _try_add(tree,factory,loss,margin,y_pred,learning_rate,regularizer):
        """try to add a specific tree and see what happens to loss"""
        newTree = loss.update_leaves(factory,margin,tree,learning_rate,regularizer)
        newPred = y_pred + factory.predict(PrunedFormula([newTree]))
        newLoss = loss.score(factory,newPred)
        return newLoss,newTree,newPred
def _inthread_try_add(trees,factory,loss,margin,y_pred,learning_rate,regularizer):
    '''in case of joblibification, use this (c)'''
    
    return [_try_add(tree,factory,loss,margin,y_pred,learning_rate,regularizer) for tree in trees]

def try_add1_bfs(allTrees,factory,learning_rate,
                 loss,breadth,y_pred,regularizer = 0.,
                 use_joblib = False,n_jobs = -1):
    '''
    select best tree to add (1 step)
    '''
    if factory.__class__ is BinaryClassificationFactory:
        y_sign = factory.labels_sign
        margin = y_sign*y_pred
    elif factory.__class__ is RegressionFactory:
        margin = factory.labels - y_pred
    else:
        raise Exception("Factory type not supported")

    if use_joblib:
        if n_jobs < 0:
            n_jobs = joblib.cpu_count() + 1 - n_jobs
        
        indices = [0]+[len(allTrees)*(i+1)/n_jobs for i in range(n_jobs)]
        treeSections = [allTrees[indices[i]:indices[i+1]] for i in range(n_jobs)]

        tasks = [joblib.delayed(_inthread_try_add)(
                    treeSection,
                    factory,
                    loss,
                    margin,
                    y_pred,
                    learning_rate,
                    regularizer) for treeSection in treeSections]
        _res = joblib.Parallel(n_jobs = n_jobs,
                               backend = "multiprocessing")(tasks)
        triples = reduce(lambda a,b:a+b, _res)

    else:
        triples = [_try_add(tree,factory,loss,margin,y_pred,learning_rate,regularizer) for tree in allTrees]   

    
    triples.sort(key = lambda el: el[0])
    



    return ([triple[1] for triple in triples[:breadth]],
            [triple[0] for triple in triples[:breadth]],
            [triple[2] for triple in triples[:breadth]])


def greed_up_features_bfs (trees,
                           factory,
                           loss,
                           learning_rate=0.25,
                           breadth=1,
                           nTrees=100,
                           trees_sample_size=100,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.,
                           use_joblib = False,
                           n_jobs = -1,
                           joblib_backend = "threading",
                           copy_pred = False,
                           initialBunch = None,
                           bias = None):
    """
    Iterative BFS over best ADD-1 results for [nTrees] iterations
    """
    allTrees = copy.copy(trees)
    if initialBunch is None:
        trees_sample = np.array(random.sample(allTrees,trees_sample_size))    
        
        if bias is None:
            bias = np.average(factory.labels,weights = factory.weights)
        
        additions,losses,preds = try_add1_bfs(trees_sample,factory,learning_rate,loss,
                                                      breadth,y_pred=bias,regularizer = regularizer)
        bunches = [PrunedFormula([_added],bias) for _added in additions]                                              
    else:
        bunches = [initialBunch]
        preds = [factory.predict(initialBunch)]
        losses = [np.sum(loss(factory,preds[0]))]
    bestScore = min(losses)

    
    if use_joblib:
        if n_jobs < 0:
            n_jobs = joblib.cpu_count()
                
        if joblib_backend == "threading":
            #create copies of data once to escape GIL forever
            factory = [factory]+[copy.deepcopy(factory) for i in range(n_jobs-1)]
            loss = [copy.deepcopy(loss) for i in range(n_jobs)]

        elif joblib_backend == "multiprocessing":
            pass
        else:
            raise ValueError, "joblib_backend must be either 'threading' or 'multiprocessing'"
    
    

    if verbose:
        print "\niteration #",0," ntrees = ", len(bunches[0]),"\nbest loss = ",bestScore
        print "learning_rate = ", learning_rate
        print "sample_size", trees_sample_size

    
    itr = 0
    while len(bunches[0]) <nTrees:

        itr+=1
        newBunches = []    
        newScores = []
        newPreds = []
        for bunch,pred in zip(bunches,preds):
            trees_sample = np.array(random.sample(allTrees,trees_sample_size))
            
            if use_joblib and joblib_backend=="threading":
                #split trees into sections
                indices = [0]+[len(trees_sample)*(i+1)/n_jobs for i in range(n_jobs)]
                treeSections = [trees_sample[indices[i]:indices[i+1]] for i in range(n_jobs)]
                if copy_pred:
                    pred = [copy.deepcopy(pred) for i in range(n_jobs)]
                else:
                    pred = [pred for i in range(n_jobs)]

                #execute sections in parallel
                tasks = [joblib.delayed(try_add1_bfs)(treeSections[ithread],factory[ithread],
                                                              learning_rate,loss[ithread],
                                                              breadth,pred[ithread],regularizer=regularizer,
                                                              use_joblib=False)
                                                    for ithread in range(n_jobs)]
                                                        
                _res = joblib.Parallel(n_jobs = n_jobs,
                               backend = "threading")(tasks)
                _additions,_losses,_preds = reduce(lambda a,b:[a[i]+b[i] for i in range(3)], _res)

                
            else:
                _additions,_losses,_preds = try_add1_bfs(trees_sample,factory,learning_rate,loss,
                                                              breadth,pred,regularizer=regularizer,
                                                              use_joblib=use_joblib,n_jobs=n_jobs)
                

            _bunches = [bunch+[_added] for _added in _additions]
            newBunches+=_bunches
            newScores += _losses
            newPreds += _preds
            
        learning_rate *= learning_rate_decay
        trees_sample_size = min(len(allTrees),trees_sample_size + trees_sample_increase)
            
        triples = zip(newScores,newBunches,newPreds)
        triples.sort(key = lambda el: el[0])
        
        newBestScore = min(newScores)
        
        if newBestScore > bestScore:
            learning_rate /=2.
            if learning_rate < 0.00001:
                break
        else: 
            bestScore = newBestScore
            bunches = [triple[1] for triple in triples[:breadth]]       
            preds = [triple[2] for triple in triples[:breadth]]       

        
        
        if verbose:
            print "\niteration #",itr," ntrees = ", len(bunches[0]),"\nbest loss = ", bestScore,"\nlast loss = ",newBestScore
            print "learning_rate = ", learning_rate
            print "sample_size", trees_sample_size       
    return bunches[0]


def wheel_up_features_bfs (initialBunch,
                           trees,
                           factory,
                           loss,
                           learning_rate=0.25,
                           nIters=100,
                           trees_sample_size=100,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.,
                           random_walk = True,
                           use_joblib = False,
                           n_jobs = -1,
                           joblib_backend = "threading",
                           copy_pred = False):
    """
    Iterative BFS over best ADD-1 results for [nTrees] iterations
    """
    allTrees = copy.copy(trees)
    
    bunch = copy.copy(initialBunch)
    pred = factory.predict(bunch)
    bestScore = sum(loss(factory,pred))
    
    if use_joblib:
        if n_jobs < 0:
            n_jobs = joblib.cpu_count()
                
        if joblib_backend == "threading":
            #create copies of data once to escape GIL forever
            factory = [copy.deepcopy(factory) for i in range(n_jobs)]
            loss = [copy.deepcopy(loss) for i in range(n_jobs)]

        elif joblib_backend == "multiprocessing":
            pass
        else:
            raise ValueError, "joblib_backend must be either 'threading' or 'multiprocessing'"
    
  
    if verbose:
        print "\niteration #",0," ntrees = ", len(bunch),"\nbest loss = ",bestScore
        print "learning_rate = ", learning_rate
        print "sample_size", trees_sample_size

    
    for itr in xrange(1,nIters+1):
        change_index= random.randint(0,len(bunch)-1) if random_walk else  (i-1)%len(bunch)
        trees_sample = random.sample(allTrees,trees_sample_size)+ [bunch[change_index]]
        bunch_wo = copy.copy(bunch)
        bunch_wo.pop(change_index)

        if use_joblib and joblib_backend=="threading":
            #split trees into sections
            indices = [0]+[len(trees_sample)*(i+1)/n_jobs for i in range(n_jobs)]
            treeSections = [trees_sample[indices[i]:indices[i+1]] for i in range(n_jobs)]
            
            pred_wo = pred - factory[0].predict([bunch[change_index]])

            if copy_pred:
                pred_wo = [copy.deepcopy(pred) for i in range(n_jobs)]
            else:
                pred_wo = [pred for i in range(n_jobs)]

            #execute sections in parallel
            tasks = [joblib.delayed(try_add1_bfs)(treeSections[ithread],factory[ithread],
                                                          learning_rate,loss[ithread],
                                                          1,pred_wo[ithread],regularizer=regularizer,
                                                          use_joblib=False)
                                                for ithread in range(n_jobs)]
                                                    
            _res = joblib.Parallel(n_jobs = n_jobs,
                           backend = "threading")(tasks)
            _additions,newScores,newPreds = reduce(lambda a,b:[a[i]+b[i] for i in range(3)], _res)
            
        else:
            pred_wo = pred - factory.predict([bunch[change_index]])

            _additions,newScores,newPreds = try_add1_bfs(trees_sample,factory,
                                                         learning_rate,loss,
                                                          1,pred_wo,regularizer=regularizer,
                                                          use_joblib=use_joblib,n_jobs=n_jobs)
        newBunches = [bunch_wo+[_added] for _added in _additions]
        

        
        learning_rate *= learning_rate_decay
        trees_sample_size = min(len(allTrees),trees_sample_size + trees_sample_increase)
            
        triples = zip(newScores,newBunches,newPreds)
        triples.sort(key = lambda el: el[0])

        newBestScore = min(newScores)
        
        if newBestScore > bestScore:
            pass
        else: 
            bestScore = newBestScore
            bunch = triples[0][1]
            bunch.insert(change_index,bunch.pop())
            pred = triples[0][2]

        
        
        if verbose:
            print "\niteration #",itr," ntrees = ", len(bunch),"\nbest loss = ", bestScore,"\nlast loss = ",newBestScore
            print "changed index",change_index
            print "learning_rate = ", learning_rate
            print "sample_size", trees_sample_size          
    return bunch


