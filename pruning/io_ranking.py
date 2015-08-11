# -*- coding: utf-8 -*-
import numpy as np
try:
    import h5py
except:
    print 'Failed to import h5py. H5 IO operations will be unavailable'

def save_as_h5(path_to_txt,output_name="mslr"):
    
    print "opening "+path_to_txt
    f = open(path_to_txt)
    labels = []
    features = []
    print "extracting..."
    for line in f:
        line = line[:line.find('#') - 1]#удалить комменты из конца линии
        ls = line.split()
        labels.append(int(ls[0]))
        features.append([float(x[x.find(':') + 1:]) for x in ls[1:]])
    f.close()
    print "converting & sorting..."
    labels = np.asarray(labels, dtype=np.int32)
    features = np.asarray(features)
    query = features[:, 0].astype(int)
    features = features[:, 1:]
    sorter = np.argsort(query)
    query,labels,features = query[sorter],labels[sorter],features[sorter]
    print "saving..."
    h5f = h5py.File(output_name, 'w')
    h5f.create_dataset('qids', data=query)
    h5f.create_dataset('labels', data=labels)
    h5f.create_dataset('features', data=features)
    h5f.close()
    print "done"
    return features,query,labels

def load_h5(name):
    print "reading from",name
    h5f = h5py.File(name,'r')
    labels = h5f['labels'][:]
    qids = h5f['qids'][:]
    features = h5f['features'][:]
    h5f.close()
    print "done"
    return features, qids, labels

def save_csv(path_to_txt,output_name = "mslr"):
    print "opening "+path_to_txt
    f = open(path_to_txt)
    labels = []
    features = []
    print "extracting..."
    for line in f:
        line = line[:line.find('#') - 1]#удалить комменты из конца линии
        ls = line.split()
        labels.append(int(ls[0]))
        features.append([float(x[x.find(':') + 1:]) for x in ls[1:]])
    f.close()
    print "converting & sorting..."
    labels = np.asarray(labels, dtype=np.int32)
    features = np.asarray(features)
    query = features[:, 0].astype(int)
    features = features[:, 1:]
    sorter = np.argsort(query)
    query,labels,features = query[sorter],labels[sorter],features[sorter]
    print "saving..."
    np.savetxt(output_name+".qids.csv",query,delimiter=',')
    np.savetxt(output_name+".labels.csv",labels,delimiter=',')
    np.savetxt(output_name+".features.csv",features,delimiter=',')
    print "done"
    return features,query,labels

def load_csv(name):
    print "reading from",name
    qids = np.loadtxt(name+".qids.csv",delimiter=',')
    labels = np.loadtxt(name+".labels.csv",delimiter=',')
    features = np.loadtxt(name+".features.csv",delimiter=',')
    print "done"
    return features, qids, labels