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

'''
def get_item_groups(items,member_index,cluster=1):
	X = items[member_index]
	estimator = KMeans(n_clusters=cluster).fit(X)
	group_center = estimator.cluster_centers_
	center_label = range(0,cluster)
	knn_clf = KNeighborsClassifier(n_neighbors=1).fit(group_center,center_label)
	_, group_index = knn_clf.kneighbors(X)
	group_members = []
	for k in range(cluster):
		group_members.append(member_index[group_index.reshape(-1)==k])
		d1_list = []
		d2_list = []
		for i in group_members[k]:
			d1,d2 = calc_dist(group_center,items[i],"cos"),calc_dist(group_center,items[i],"l2")
			d1_list.append(d1)
			d2_list.append(d2)
		print(np.percentile(d1_list,[0,50,75,95,100]))
		print(np.percentile(d2_list,[0,50,75,95,100]))
	return group_center,group_members
'''
from collections import defaultdict

def get_item_groups(items,member_index):
	group_center = [np.mean(items[member_index],axis=0)]
	group_member = [member_index]
	return group_center,group_member

def build_index(items,tags):
	tag_set = set(tags.reshape(-1))
	group_centers = []
	group_indices = []
	group_tags = []
	loop_id = 0
	member_indices = defaultdict(list)
	for i in range(len(tags)):
		for tag in tags[i]:
			member_indices[tag].append(i)
	for tag in tag_set:
		#member_index = [] # the index of item with a given tag
		#for i in range(len(tags)):
		#	if tag in tags[i]:
		#		member_index.append(i) # Too slow
		member_index = np.array(member_indices[tag])
		group_center,group_member = get_item_groups(items,member_index)
		# Now each tag has only one group, as default
		group_centers += group_center
		group_indices += group_member
		group_tags += [tag]*len(group_center)
		print("Process %d/%d" % (loop_id+1,len(tag_set)))
		loop_id += 1
	return group_centers,group_indices,group_tags

items,users,tags,neighbors = bcutils.load_input_data()

build_index(items,tags)

