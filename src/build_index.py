import argparse
import time

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

import metrics
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
		#print("Process %d/%d" % (loop_id+1,len(tag_set)))
		loop_id += 1
	return group_centers,group_indices,group_tags



def softmax_dict_values(x:dict):
	sval = sum(np.exp(list(x.values())))
	return {k:np.exp(x[k])/sval for k in x}

def regular_dict_values(x:dict):
	sval = max(sum(list(x.values())),1e-6)
	return {k:x[k]/sval for k in x}

def search_first_level(users,group_centers,group_indices,group_tags,num_group=10,num_candidate=1000):
	candidate_groups = []
	center_labels = np.arange(0,len(group_centers))
	knn_clf = KNeighborsClassifier(n_neighbors=num_group).fit(group_centers,center_labels)
	_, prefer_groups = knn_clf.kneighbors(users)
	for i in range(len(users)):
		candidates = dict()
		for group_idx in prefer_groups[i]:
			candidates[group_idx] = calc_dist(users[i],group_centers[group_idx])
			#candidates[index] = 1
		candidates = regular_dict_values(candidates)
		#candidates = softmax_dict_values(candidates)
		candidates = {k:int(v*num_candidate) for k,v in candidates.items()}
		candidate_groups.append(candidates)
	return candidate_groups

def transform_to_results(raw_results):
	"""
	Convert format of data
	X: [[index0,index1,...],[,...],...]
	results: [(float,[(index0,dist0),(index1,dist1),...]),(float,list),...]
	"""
	return [[0,[(e,0) for e in item]] for item in raw_results]

def get_random_neighbors(items,users,neighbor_count):
	results = []
	print("Start to random selection")
	t0 = time.time()
	index_set = np.arange(items.shape[0])
	for user in users:
		results.append(np.random.choice(index_set,neighbor_count,replace=False))
	print("Random selection in",time.time()-t0,"seconds")
	return np.array(results)

def search_by_ann_benchmark(items,users,tags,neighbors,neighbor_count=100):
	# base conf
	definition = Definition(
		algorithm="hnswlib", # 
		docker_tag=None, # not needed
		module="ann_benchmarks.algorithms.hnswlib",
		constructor="HnswLib",
		arguments=["angular", {"M": 8,"efConstruction": 500}],
		query_argument_groups=[[200]],
		disabled=False
	)
	algo = instantiate_algorithm(definition)
	distance = "angular"
	run_count = 1

	try:
		prepared_queries = False
		if hasattr(algo, "supports_prepared_queries"):
				prepared_queries = algo.supports_prepared_queries()
		t0 = time.time()
		memory_usage_before = algo.get_memory_usage()
		algo.fit(items)
		build_time = time.time() - t0
		index_size = algo.get_memory_usage() - memory_usage_before
		print('Built index in', build_time, "seconds")
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
						algo, items, users, distance, neighbor_count, run_count, False)
				descriptor["build_time"] = build_time
				descriptor["index_size"] = index_size
				descriptor["algo"] = definition.algorithm
				bcutils.save_object([descriptor,results],"results_hnsw_0802.obj")
				descriptor["diversity"] = metrics.diversity(results, tags)
				descriptor["coverage"] = metrics.coverage(results, neighbors)
				baseline_neighbors = transform_to_results(neighbors)
				descriptor["baseline_diversity"] = metrics.diversity(baseline_neighbors, tags)
				descriptor["baseline_coverage"] = metrics.coverage(baseline_neighbors, neighbors)
				random_neighbors = get_random_neighbors(items,users,neighbor_count)
				random_neighbors = transform_to_results(random_neighbors)
				descriptor["random_diversity"] = metrics.diversity(random_neighbors, tags)
				descriptor["random_coverage"] = metrics.coverage(random_neighbors, neighbors)
				#print(descriptor)
	finally:
		algo.done()
	return results,descriptor

def collect_candidates(items,users,candidate_groups,group_indices):
	for i in range(len(users)):
		for group_idx,num_item in candidate_groups.items():
			#search users[i] in items[group_indices[group_idx]] out num_item
			pass
	return # collective candidates

	# then second level

def test_descriptor(items,users,tags,neighbors,neighbor_count=100):
	descriptor,results = bcutils.load_object("results_hnsw_0802.obj")
	descriptor["diversity"] = metrics.diversity(results, tags)
	descriptor["coverage"] = metrics.coverage(results, neighbors)
	baseline_neighbors = transform_to_results(neighbors)
	descriptor["baseline_diversity"] = metrics.diversity(baseline_neighbors, tags)
	descriptor["baseline_coverage"] = metrics.coverage(baseline_neighbors, neighbors)
	random_neighbors = get_random_neighbors(items,users,neighbor_count)
	random_neighbors = transform_to_results(random_neighbors)
	descriptor["random_diversity"] = metrics.diversity(random_neighbors, tags)
	descriptor["random_coverage"] = metrics.coverage(random_neighbors, neighbors)
	return results,descriptor

if __name__=="__main__":
	items,users,tags,neighbors = bcutils.load_input_data()
	#results,descriptor = search_by_ann_benchmark(items,users)
	results,descriptor = test_descriptor(items,users,tags,neighbors)
	print(descriptor)
	'''
	#group_centers,group_indices,group_tags = build_index(items,tags)
	#bcutils.save_object([group_centers,group_indices,group_tags],"group0802.obj")
	group_centers,group_indices,group_tags = bcutils.load_object("group0802.obj")
	print("Build a total of",len(group_centers),"groups.")
	candidate_groups = search_first_level(users,group_centers,group_indices,group_tags,num_group=5)
	print(candidate_groups[:10])
	'''



