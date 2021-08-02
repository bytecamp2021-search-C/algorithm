import argparse
import json
import logging
import os
import threading
import time
import traceback

import colors
import docker
import numpy
import psutil

from ann_benchmarks.algorithms.definitions import (Definition,instantiate_algorithm)
from ann_benchmarks.datasets import get_dataset, DATASETS, get_dataset_fn
from ann_benchmarks.distance import metrics, dataset_transform
from ann_benchmarks.results import store_results
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import h5py
from ann_benchmarks.runner import run, run_individual_query
from sklearn.neighbors import KNeighborsClassifier

# step1 build dataset
# python3 create_dataset.py --dataset=glove-25-angular
# generate tags
def generate_tags(X, cluster, tag_count):
    # use kmeans to get cluster center
    estimator = KMeans(n_clusters=cluster)
    estimator.fit(X)
    # get center and center label
    cluster_center = estimator.cluster_centers_
    center_label=range(0,cluster)
    # use knn to generate tags
    knn_clf=KNeighborsClassifier(n_neighbors=tag_count)
    knn_clf.fit(cluster_center,center_label)
    _, tags=knn_clf.kneighbors(X)
    return tags

# tag_num: all tag number
# tag_count : for element
def generate_dataset_tags(data_set, tag_num, tag_count):
    D, _ = get_dataset(dataset)
    if "tags" not in D.keys():
        X_train = numpy.array(D['train'])
        X_test = numpy.array(D['test'])
        X_train, X_test = dataset_transform(D)
        print("x train size %d" % len(X_train))
        X = numpy.vstack((X_train,X_test))
        print("start kmeans data size: %d cluster %d " % (len(X), tag_num))
        tags = generate_tags(X, tag_num, tag_count)
        Train_tags = numpy.split(tags, [len(X_train)])[0]
        D.close()
        # store tags
        hdf5_fn = get_dataset_fn(data_set)
        hdf5_f = h5py.File(hdf5_fn, 'a')
        f_tags = hdf5_f.create_dataset('tags', (len(X_train), tag_count), dtype='i')
        for i, x in enumerate(Train_tags):
            f_tags[i] = x
        hdf5_f.close()
    else:
        D.close()
    
def diversity(candidates, tags):
    tag_c = 0
    for _, items in candidates:
        tag_count = {}
        for item in items:
            tag = tags[item[0]]
            for e in tag:
                if e in tag_count.keys():
                    tag_count[e] = tag_count[e] + 1
                else:
                    tag_count[e] = 1
        tag_c += len(tag_count)
    return tag_c/len(candidates)

def baseline_diversity(neighbors, tags):
    tag_c = 0
    for i in range(len(neighbors)):
        neighbor = neighbors[i]
        tag_count = {}
        for item_id in neighbor:
            tag = tags[item_id]
            for e in tag:
                if e in tag_count.keys():
                    tag_count[e] = tag_count[e] + 1
                else:
                    tag_count[e] = 1
        tag_c += len(tag_count)
    return tag_c/len(neighbors)

def coverage(results, neighbors):
    coverage = 0.0
    for i, candidate in enumerate(results):
        neighbor = neighbors[i]
        match = 0
        for item in candidate[1]:
            #print(item[0])
            if item[0] in neighbor:
                match = match + 1
        coverage = coverage + match/len(neighbor)
    return coverage/len(results)

def get_random_neighbors(X_train,neighbors):
	results = []
	index_set = np.arange(X_train.shape[0])
	for items in neighbors:
		results.append(np.random.choice(index_set,neighbors.shape[1]))
	return np.array(results)

# base conf
definition = Definition(
        algorithm="hnswlib", # 
        docker_tag=None,    # not needed
        module="ann_benchmarks.algorithms.hnswlib",
        constructor="HnswLib",
        arguments=["angular", {"M": 8,    "efConstruction": 500}],
        query_argument_groups=[[200]],
        disabled=False
)
dataset="glove-25-angular"
count=100
run_count = 1

# generate tags for item
generate_dataset_tags(dataset, 10000, 5)

# init class
algo = instantiate_algorithm(definition)

# load data
D, dimension = get_dataset(dataset)
X_train = numpy.array(D['train'])
X_test = numpy.array(D['test'])
tags = numpy.array(D['tags'])
distance = D.attrs['distance']
neighbors = numpy.array(D['neighbors'])
print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
print('got %d queries' % len(X_test))
X_train, X_test = dataset_transform(D)

# check model
try:
    prepared_queries = False
    if hasattr(algo, "supports_prepared_queries"):
            prepared_queries = algo.supports_prepared_queries()

    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before
    print('Built index in', build_time)
    print('Index size: ', index_size)

    query_argument_groups = definition.query_argument_groups
    # Make sure that algorithms with no query argument groups still get run
    # once by providing them with a single, empty, harmless group
    if not query_argument_groups:
            query_argument_groups = [[]]

    for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." %
                        (pos, len(query_argument_groups)))
            if query_arguments:
                    algo.set_query_arguments(*query_arguments)
            descriptor, results = run_individual_query(
                    algo, X_train, X_test, distance, count, run_count, False)
            descriptor["build_time"] = build_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = definition.algorithm
            descriptor["dataset"] = dataset
            print(type(results),len(results),len(results[0]),len(results[0][1]))
            #print([results[k] for k in range(10)])
            #print([results[k][0] for k in range(10)])
            print(type(tags),len(tags),len(tags[0]))
            print(type(neighbors),len(neighbors),len(neighbors[0]))
            descriptor["diversity"] = diversity(results, tags)
            descriptor["baseline_diversity"] = baseline_diversity(neighbors, tags)
            random_neighbors = get_random_neighbors(X_train,neighbors)
            descriptor["random_diversity"] = baseline_diversity(random_neighbors, tags)
            descriptor["coverage"] = coverage(results, neighbors)
            print(descriptor)
finally:
    algo.done()