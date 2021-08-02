import argparse

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

import bcutils

def calc_dist(x_,y_,metric="cos"):
	x = x_.reshape(-1)
	y = y_.reshape(-1)
	if metric=="cos":
		return np.dot(x,y)/np.sqrt(np.dot(x,x))/np.sqrt(np.dot(y,y))
	elif metric=="l2":
		return np.sqrt(np.dot(x-y,x-y))
	else:
		print("[!] unrecognized metric.")
		return 0.0

def get_item_groups(items,member_index,cluster=1):
	X = items[member_index]
	estimator = KMeans(n_clusters=cluster).fit(X)
	group_center = estimator.cluster_centers_
	center_label = range(0,cluster)
	knn_clf = KNeighborsClassifier(n_neighbors=1).fit(group_center,center_label)
	_, group_members = knn_clf.kneighbors(X)
	for k in range(cluster):
		for i in group_members:
			print(calc_dist(group_center,items[i],"cos"),calc_dist(group_center,items[i],"l2"))
	return group_center,group_members

def build_index(items,tags):
	tag_set = set(tags.reshape(-1))
	for tag in tag_set:
		member_index = [] # the index of item with a given tag
		for i in range(len(tags)):
			if tag in tags[i]:
				member_index.append(i)
		group_center,group_members = get_item_groups(items,member_index)

items,users,tags,neighbors = bcutils.load_input_data()

build_index(items,tags)

