# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:07:02 2015
@author: ayanami
"""
import numpy as np

    
def dcg(relevances, rank=None):
    """Discounted cumulative gain"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)
 
def single_dcg(arg):
    """at one point"""
    i, label = arg
    return (2 ** label - 1) / np.log2(i + 2)
 
def ndcg(relevances, rank=None):
    """Normalized DGC"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg
def mean_ndcg(y_true, y_pred, query_ids, rank=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    query_ids = np.asarray(query_ids)
    # assume query_ids are sorted
    ndcg_scores = []
    previous_qid = query_ids[0]
    previous_loc = 0
    for loc, qid in enumerate(query_ids):
        if previous_qid != qid:
            chunk = slice(previous_loc, loc)
            ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
            ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
            previous_loc = loc
        previous_qid = qid

    chunk = slice(previous_loc, loc + 1)
    ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
    ndcg_scores.append(ndcg(ranked_relevances, rank=rank))
    return np.mean(ndcg_scores)


def roc_auc_score(Ytrue,Yscored):
    Ytrue = np.array(Ytrue)
    Yscored = np.array(Yscored)    
    
    #unique Y values in ascension order
    Yunique = np.unique(Ytrue)
    
    #sort the sequence
    order = np.array(zip(-Ytrue,Yscored), dtype=[('-Yt', 'float'), ('Ys', 'float')])
    indices = order.argsort(order = ('Ys','-Yt'))
    Ytrue, Yscored = Ytrue[indices],Yscored[indices]
    
    #counts of processed samples for every possible Y rank
    counts = {y:0 for y in  Yunique}
    
    pairs_in_order = 0    
    all_pairs = 0
    for i in xrange(len(Ytrue)):
        """once i'll become loop-free"""
        ytrue = Ytrue[i]
        YcorrectlyPlaced = Yunique[:np.where(Yunique==ytrue)[0]]
        #since Ysc[other] < Ysc, all the pairs with Ytr[other] < Ytr are placed correctly
        
        for ytrue_other in YcorrectlyPlaced:
            pairs_in_order += counts[ytrue_other]
        all_pairs +=i-counts[ytrue]
        
        counts[ytrue]+=1
    return pairs_in_order/float(all_pairs)