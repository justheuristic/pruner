import numpy as np
from sklearn.externals import joblib
from copy import deepcopy
from greedy import greed_up_features_bfs
import random

#metrics of usability
def usability_entropy(formula,*factories):
    ans = 0
    for factory in factories:
        for Y in formula.staged_predict(factory):
            var = np.var(Y)
            ans += -var*np.log2(var) if var != 0 else 0
    return ans

from itertools import izip
def normalized_usability_entropy(formula,*factories):
    ans = 0#var * log(var_normed)?
    for Ys in izip(*[formula.staged_predict(fct) for fct in factories]):
        variances = [np.var(Y) for Y in Ys]
        norm = sum(variances)
        if norm !=0:
            for var in variances:
                var_normed = var/norm
                ans += - var_normed*np.log2(var_normed) if var_normed != 0 else 0
    return ans

class Penalized_entropy:
    def __init__(self,metric,pow = 1):
        self.metric = metric#assert that metric is strictly > 0 and the_less_the_better
        self.pow = pow
    def __call__(self,formula,*factories):
        ans = self.metric(formula,*factories)
        bin_sizes = np.array([fct.events.shape[0] for fct in factories],dtype = "float64")
        bin_sizes /= np.sum(bin_sizes)
        entropy = -sum( (p* np.log2(p) if p != 0 else 0) for p in bin_sizes)
        return ans/(entropy**self.pow) #penalize for unballanced splits
    
penalized_usability_entropy = Penalized_entropy(usability_entropy,1)
penalized_normalized_entropy = Penalized_entropy(normalized_usability_entropy,1)

from copy import deepcopy
def get_split_scores(factory,thresholds,formula,
                     metric = None,#p.e. usability entropy
                     use_joblib = False,
                     joblib_backend = 'threading',
                     n_jobs = -1,
                     min_events_fraction_leaf = 0.,verbose = False):

    if metric == None:
        metric = penalized_usability_entropy
    if min_events_fraction_leaf <=1:
        min_events_fraction_leaf = int(min_events_fraction_leaf*sum(factory.weights))
    if verbose:
        print min_events_fraction_leaf, sum(factory.weights)

    if not use_joblib:
        scores = np.repeat(float("inf"),len(thresholds))
        for i,(feature,cut,_) in enumerate(thresholds):
            predicate =  (factory.events[:,feature] > cut)

            #skip the edge cases... (inf penalty)
            if np.all(predicate) or (not np.any(predicate)):
                #if this split does not split, fuggedaboutit
                continue 
            if min_events_fraction_leaf>0:
                #get rid of too uneven a cuts
                sum_weight = np.sum(factory.weights)
                true_weight = np.sum(factory.weights[predicate])
                false_weight = sum_weight - true_weight
                if true_weight < min_events_fraction_leaf or false_weight < min_events_fraction_leaf:
                    if verbose: print "t:",true_weight,"f:",false_weight, "discarded"
                    continue
                if verbose: print "t:",true_weight,"f:",false_weight, "passed"
            #compute score
            subFactories = factory.split_by(predicate)
            scores[i] = metric(formula,*subFactories)
    else:
        if n_jobs < 0:
            n_jobs = joblib.cpu_count() +1 - n_jobs
       
        indices = [0]+[len(thresholds)*(i+1)/n_jobs for i in range(n_jobs)]
        thresholdSections = [thresholds[indices[i]:indices[i+1]] for i in range(n_jobs)]
        
        if joblib_backend == 'threading':
            factory = [deepcopy(factory) for i in range(n_jobs)]
            formula = [deepcopy(formula) for i in range(n_jobs)]
            metric = [deepcopy(metric) for i in range(n_jobs)] #in case it has some internal data
            
            jobs = (joblib.delayed(get_split_scores)(factory[i],thresholdSection, formula[i],
                                                 metric=metric[i],use_joblib = False,
                                                 min_events_fraction_leaf = min_events_fraction_leaf,
                                                 verbose = verbose)
                                    for i,thresholdSection in enumerate(thresholdSections))
        else:
            jobs = (joblib.delayed(get_split_scores)(factory,thresholdSection, formula,
                                                 metric=metric,use_joblib = False,
                                                 min_events_fraction_leaf = min_events_fraction_leaf,
                                                 verbose = verbose)
                                    for thresholdSection in thresholdSections)
        scores = np.hstack(joblib.Parallel(n_jobs = n_jobs, backend = joblib_backend)(jobs))
    return scores

class hierarchyNode:
    def __init__(self,factory):
        ''' i am a non-oblivious higher-order tree'''

        self.isLeaf = True
        self.child = {}
        self.factory = factory#note that they are removed once used in the create_hierarchy function
    def split(self,factory,index_array = None,*args,**kwargs):
        #what is index_array?: it is just an array of sample indexes used to recall where do
        #samples go when predicting for arbitrary new data

        if self.isLeaf:
            return factory if index_array is None else (factory,index_array)
        else:
            feature,cut,_ = self.dichotomy
            predicate = factory.events[:,feature]>cut
            subFactories = factory.split_by(predicate,*args,indices = index_array,**kwargs) 
            #first predicate==0, than predicate==1
            
            if index_array is not None:
                richDict = {str(i):self.child[i].split(fct,index_array = ind) for i,(fct,ind) in enumerate(subFactories)}
            else:
                richDict = {str(i):self.child[i].split(fct) for i,fct in enumerate(subFactories)}
            #richDict is at most depth-2 dict. Leaves are of depth 1, furtherly splitted nodes are of depth 2
            
            flatDict = {}
            for key,value in richDict.items():
                if type(value) is dict:
                    for v_key,v_value in value.items():
                        flatDict[str(key)+str(v_key)] = v_value
                else:
                    flatDict[str(key)] = value
                    
            
            
            return flatDict

        import random
from greedy import PrunedFormula as pf
def create_hierarchy(factory,
                 formula,
                 thresholds,
                 max_depth = 3,
                 min_events_split = 0.125,
                 min_events_leaf = 0.0,
                 event_sample= 0.2,
                 tree_sample = 0.1,
                 threshold_sample = 1,
                 verbose = True,
                 *args, **kwargs):
    """I create a tree of hierarchy nodes with given parameters;
    Note that if the parameters are less than 1, they are considered
    as fractions of the whole factory, all trees, thresholds, etc."""
    #converting parameters into nominal form if they are provided as shares.
    if event_sample <=1:
        event_sample = int(event_sample*np.sum(factory.weights))
    if min_events_split <=1:
        min_events_split = int(min_events_split*np.sum(factory.weights))
    if min_events_leaf <= 1:
        min_events_leaf =int(min_events_leaf*np.sum(factory.weights))
    if tree_sample <=1:
        tree_sample = int(tree_sample*len(formula))
    if threshold_sample <= 1:
        threshold_sample = int(threshold_sample*len(thresholds))
    
    #initializing tree
    root = hierarchyNode(factory)
    node_pool = [root]
    
    for depth in xrange(max_depth):
        next_pool = []
        for node in node_pool:
            
            #drawing samples:
            #note that subsamples are resampled on every step.
            #this allows to stick with fixed sample subset without
            #loosing accuracy at lower tree levels
            n_events = node.factory.events.shape[0]
            event_selector = np.random.choice(np.arange(n_events),
                                              min(event_sample,n_events),replace = False)
            sample_factory = node.factory.select_by(event_selector)
            
            
            sample_thresholds = random.sample(thresholds,min(threshold_sample,len(thresholds)))
            sample_trees = pf(random.sample(formula,min(tree_sample,len(formula))))
            
            sum_node_weights = np.sum(node.factory.weights)
            
            if min_events_leaf >= 0.5*sum_node_weights:
                if verbose:
                    print 'Alarm! min_events_leaf restriction too high to find anything'
                del node.factory
                continue #node stays a leaf.            
                
            #computing optimal split
            min_sample_fraction_leaf = np.sum(sample_factory.weights)*(min_events_leaf*1./sum_node_weights)
            print min_sample_fraction_leaf, np.sum(sample_factory.weights)
            
            scores = get_split_scores(sample_factory,sample_thresholds,sample_trees,*args,
                                      min_events_fraction_leaf = min_sample_fraction_leaf,
                                      **kwargs)
            
            if np.isinf(np.min(scores)):
                if verbose:
                    print 'Alarm! No suitable dichotomy was found for the node'
                del node.factory
                continue #node stays a leaf.
                
            node.dichotomy = feature,cut,_= sample_thresholds[np.argmin(scores)]
            
            predicate = node.factory.events[:,feature] > cut
            subFactories = node.factory.split_by(predicate) #first negative, than positive
            
            for i,fct in enumerate(subFactories):
                node.child[i] = hierarchyNode(fct)
                
                if verbose: 
                    print "leaf samples:",fct.events.shape[0],"/",min_events_split
                if fct.events.shape[0] <min_events_leaf:
                    print scores
                    print node.dichotomy
                    raise ValueError
                    
                if fct.events.shape[0] >= min_events_split and depth < max_depth -1: 
                    #if leaf is to be split, add it to the algorithm's "todo" list
                    next_pool.append(node.child[i])
                else:
                    del node.child[i].factory #else dispose of the samples as we won't need them anymore
                
            del node.factory
            node.isLeaf = False
            
        if verbose: print "depth:",depth,",nodes in the next pool:",len(next_pool)
        node_pool = next_pool#preparing to expand the next depth level
        
    return root
            
        
class SplitPrunedFormula(dict):
    '''Output of train_splitted_bosts;
    a dict with criteria field'''
    def __init__(self,splits,treeRoot):
        self.treeRoot = treeRoot
        dict.__init__(self,splits)
    def predict(self,factory):
        """predict with a split-pruned formula"""
        
        factIndDict = self.treeRoot.split(factory,np.arange(factory.events.shape[0]))
        factories = {i:factIndDict[i][0] for i in factIndDict}
        indices = {i:factIndDict[i][1] for i in factIndDict}
        
        y_pred = np.zeros(factory.n_events)
        for leaf in factories.keys():
            y_pred[indices[leaf]] = self[leaf].predict(factories[leaf])
        return y_pred
    def staged_predict(self,factory):
        factIndDict = self.treeRoot.split(factory,np.arange(factory.events.shape[0]))
        factories = {i:factIndDict[i][0] for i in factIndDict}
        indices = {i:factIndDict[i][1] for i in factIndDict}
        
        y_pred = np.zeros(factory.n_events)
        staged_predictors = {leaf: self[leaf].staged_predict(factories[leaf]) for leaf in factories.keys()}
        while len(staged_predictors) != 0:
            for leaf in staged_predictors.keys():
                try:
                    y_pred[indices[leaf]] = staged_predictors[leaf].next()
                except:
                    del staged_predictors[leaf]
            yield y_pred
            

def train_splitted_boosts( trees,
                           factory,
                           hierarchy_root,
                           loss,
                           learning_rate=0.25,
                           breadth=1,
                           nTrees_leaf=10,
                           trees_sample_size=100,
                           verbose = True,
                           learning_rate_decay = 1.,
                           trees_sample_increase = 0,
                           regularizer = 0.,
                           weights_outside_leaf = 0.,
                           inclusion_outside_leaf = 0.,
                           use_joblib = False,
                           joblib_backend = "threading",
                           use_joblib_inner = False,
                           joblib_backend_inner = "threading",
                           copy_pred_inner = False,
                           n_jobs = -1,
                           n_jobs_inner=-1,
                           initialTrees= None):
    """
    make greedy prune for every leaf in split. I know i should be with kwargs
    """

    factories = hierarchy_root.split(factory,
                                       offclass_weights = weights_outside_leaf,
                                       offclass_sample= inclusion_outside_leaf)
    
    leaves = factories.keys()
    if initialTrees == None:
        initialTrees = {leaf:None for leaf in leaves}
    
        
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
                               joblib_backend = joblib_backend_inner,
                               copy_pred = copy_pred_inner,
                               initialBunch = initialTrees[leaf]
                               )

            classis.append(classi)
    else: #use joblib
        if joblib_backend not in ["threading","multiprocessing"]:
            raise ValueError, "please specify a valid backend type: threading or multiprocessing"
        tasks = [joblib.delayed(greed_up_features_bfs)(
                               deepcopy(trees),
                               factories[leaf],
                               deepcopy(loss),
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
                               joblib_backend = joblib_backend_inner,
                               copy_pred = copy_pred_inner,
                               initialBunch = deepcopy(initialTrees[leaf])#just in case it's the same...
                               )for leaf in leaves]
                               
        
        classis = joblib.Parallel(n_jobs = n_jobs,
                                  backend = "threading" if joblib_backend == "threading" else "multiprocessing",
                                  verbose=verbose)(tasks)

    
    return SplitPrunedFormula({leaves[i]:classis[i] for i in range(len(leaves))},hierarchy_root)