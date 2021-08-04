import time

from lib.ann.ann_benchmarks.algorithms.definitions import (Definition, instantiate_algorithm)
from lib.ann.ann_benchmarks.distance import metrics
import numpy as np
from lib.ann.ann_benchmarks.runner import run_individual_query
from sklearn.neighbors import KNeighborsClassifier

import bcmetrics
from utils import bcutils


def calc_dist(x_, y_, metric="cos"):
    x = x_.reshape(-1)
    y = y_.reshape(-1)
    if metric == "cos":
        return np.dot(x, y) / np.sqrt(np.dot(x, x)) / np.sqrt(np.dot(y, y))
    elif metric == "l2":
        return np.sqrt(np.dot(x - y, x - y))
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


def get_item_groups(items, member_index):
    group_center = [np.mean(items[member_index], axis=0)]
    group_member = [member_index]
    return group_center, group_member


def build_index(items, tags):
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
        # member_index = [] # the index of item with a given tag
        # for i in range(len(tags)):
        #	if tag in tags[i]:
        #		member_index.append(i) # Too slow
        member_index = np.array(member_indices[tag])
        group_center, group_member = get_item_groups(items, member_index)
        # Now each tag has only one group, as default
        group_centers += group_center
        group_indices += group_member
        group_tags += [tag] * len(group_center)
        # print("Process %d/%d" % (loop_id+1,len(tag_set)))
        loop_id += 1
    return group_centers, group_indices, group_tags


def softmax_dict_values(x: dict):
    sval = sum(np.exp(list(x.values())))
    return {k: np.exp(x[k]) / sval for k in x}


def regular_dict_values(x: dict):
    sval = max(sum(list(x.values())), 1e-6)
    return {k: x[k] / sval for k in x}


def search_first_level(users, group_centers, group_indices, group_tags, num_group=10, num_candidate=1000):
    candidate_groups = []
    center_labels = np.arange(0, len(group_centers))
    knn_clf = KNeighborsClassifier(n_neighbors=num_group).fit(group_centers, center_labels)
    t0 = time.time()
    _, prefer_groups = knn_clf.kneighbors(users)
    for i in range(len(users)):
        candidates = dict()
        for group_idx in prefer_groups[i]:
            candidates[group_idx] = calc_dist(users[i], group_centers[group_idx])
        # candidates[index] = 1
        candidates = regular_dict_values(candidates)
        # candidates = softmax_dict_values(candidates)
        candidates = {k: max(int(v * num_candidate), 1) for k, v in candidates.items()}
        candidate_groups.append(candidates)
    total_time = time.time() - t0
    return candidate_groups, total_time


def transform_to_results(raw_results):
    """
    Convert format of data
    X: [[index0,index1,...],[,...],...]
    results: [(float,[(index0,dist0),(index1,dist1),...]),(float,list),...]
    """
    return [[0, [(e, 0) for e in item]] for item in raw_results]


def get_random_neighbors(items, users, neighbor_count):
    results = []
    print("Start to random selection")
    t0 = time.time()
    index_set = np.arange(items.shape[0])
    for user in users:
        results.append(np.random.choice(index_set, neighbor_count, replace=False))
    print("Random selection in", time.time() - t0, "seconds")
    return np.array(results)


def search_by_ann_benchmark(items, users, tags, neighbors, neighbor_count=100, evaluate=False):
    # base conf
    definition = Definition(
        algorithm="hnswlib",  #
        docker_tag=None,  # not needed
        module="ann_benchmarks.algorithms.hnswlib",
        constructor="HnswLib",
        arguments=["angular", {"M": 8, "efConstruction": 500}],
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
            # bcutils.save_object([descriptor,results],"./model/results_hnsw_0802.obj")

            if evaluate is True:
                descriptor["diversity"] = bcmetrics.diversity(results, tags)
                descriptor["coverage"] = bcmetrics.coverage(results, neighbors)

                baseline_neighbors = transform_to_results(neighbors)
                descriptor["baseline_diversity"] = bcmetrics.diversity(baseline_neighbors, tags)
                descriptor["baseline_coverage"] = bcmetrics.coverage(baseline_neighbors, neighbors)

                random_neighbors = get_random_neighbors(items, users, neighbor_count)
                random_neighbors = transform_to_results(random_neighbors)
                descriptor["random_diversity"] = bcmetrics.diversity(random_neighbors, tags)
                descriptor["random_coverage"] = bcmetrics.coverage(random_neighbors, neighbors)

    finally:
        algo.done()
    return results, descriptor


def search_by_ann_benchmark_personalized(items, users, tags, neighbors, num_counts):
    # base conf

    # return algo ???

    results = []
    for user_id in range(len(users)):
        # print("Running query %d of %d..." % (user_id, len(users)))
        # Close an output
        # Consider modify runner.py
        des, res = run_individual_query(
            algo, items, users[user_id:user_id + 1], distance, num_counts[user_id], run_count, False)
        des["index_size"] = index_size
        des["algo"] = definition.algorithm
        results += res
    return results


'''
def collect_candidates(items,users,candidate_groups,group_indices,search_mode="ann_benchmarks"):
	group_attracted_users = dict()
	for i in range(len(candidate_groups)):
		for group_idx,num_item in candidate_groups[i].items():
			if group_idx not in group_attracted_users:
				group_attracted_users[group_idx] = [[],[]]
			group_attracted_users[group_idx][0].append(i)
			group_attracted_users[group_idx][1].append(num_item)

	cumulate_time = 0
	collective_candidates = [[] for user in users]
	pos = 0
	for group_idx,targets in group_attracted_users.items():
		if search_mode=="ann_benchmarks":
			#search users[i] in items[group_indices[group_idx]] out num_item
			t0 = time.time()
			res = search_by_ann_benchmark_personalized(items[group_indices[group_idx]],\
				users[targets[0]],tags[group_indices[group_idx]],\
				neighbors[targets[0]],targets[1])
			for i in range(len(targets[0])):
				user_id = targets[0][i]
				collective_candidates[user_id] += res[i]
		print("Finished group #",pos+1," time cost:",time.time()-t0)
		cumulate_time += (time.time()-t0)
		pos += 1
	print("Time Cost:",cumulate_time)
	return collective_candidates,cumulate_time
'''


def build_hnsw_index_second_level(items, group_indices):
    group_algos = []
    definition = Definition(
        algorithm="hnswlib",  #
        docker_tag=None,  # not needed
        module="ann_benchmarks.algorithms.hnswlib",
        constructor="HnswLib",
        arguments=["angular", {"M": 8, "efConstruction": 500}],
        query_argument_groups=[[200]],
        disabled=False
    )
    for group_idx in range(len(group_indices)):
        indices = group_indices[group_idx]
        algo = instantiate_algorithm(definition)

        prepared_queries = False
        if hasattr(algo, "supports_prepared_queries"):
            prepared_queries = algo.supports_prepared_queries()
        t0 = time.time()
        # memory_usage_before = algo.get_memory_usage()
        algo.fit(items[indices])
        build_time = time.time() - t0
        # index_size = algo.get_memory_usage() - memory_usage_before
        print('Built index for group #', group_idx, 'in', build_time, "seconds")
        query_arguments = definition.query_argument_groups[0]

        algo.set_query_arguments(*query_arguments)
        group_algos.append(algo)
    return group_algos


'''
def greedy_selection(items,users,tags,results,num_reserve=100):
	final_results = []
	t0 = time.time()
	for user_id in range(len(users)):
		user = users[user_id]
		indices = list(results[user_id].keys())
		weights = [results[user_id][k] for k in indices]
		# object_function = coverage + diversity

		candidates = []
		#for i in range(num_reserve):
		#	for j in range()
		candidates = sorted([[indices[i],weights[i]] for i in range(len(indices))],key=lambda x:1-x[1])
		candidates = [candidates[i][0] for i in range(num_reserve)]
		final_results.append(candidates)
	total_time = time.time()-t0
	return final_results,total_time
'''


def single_query_hnsw(user, sub_items, count, algo, distance="angular"):
    start = time.time()
    candidates = algo.query(user, count)
    total = (time.time() - start)
    if distance:
        candidates = [(int(idx), float(metrics[distance]['distance'](user, sub_items[idx])))
                      for idx in candidates]
    else:
        candidates = [int(idx) for idx in candidates]
    return candidates


def search_second_level(items, users, candidate_groups, group_indices, group_algos, num_candidate=1000,
                        num_reserve=100):
    prepared_queries = False
    best_search_time = float('inf')
    results = []
    distance = None
    t0 = time.time()
    for user_id in range(len(users)):
        candidates = dict() if distance else list()
        # expect_num = 0
        for group_idx, num_cnt in candidate_groups[user_id].items():
            algo = group_algos[group_idx]
            item_id = group_indices[group_idx]
            # expect_num += num_cnt
            # res = single_query(users[user_id],num_candidate-expect_num,algo)
            res = single_query_hnsw(users[user_id], items[item_id], num_cnt, algo, distance)
            # res = mydpp(num_cnt,users[user_id],items[item_id])
            if distance:
                for e in res:
                    candidates[item_id[e[0]]] = e[1]
            else:
                candidates += res
        if distance:
            indices = list(candidates.keys())
            results.append(sorted(indices, key=lambda x: candidates[x])[:num_reserve])
        else:
            results.append(np.random.choice(candidates, num_reserve, replace=False).tolist())
    total_time = time.time() - t0
    print("Time Cost:", total_time)
    return results, total_time


if __name__ == "__main__":
    items, users, tags, neighbors = bcutils.load_input_data()
    # results,descriptor = search_by_ann_benchmark(items,users)
    # results,descriptor = test_descriptor(items,users,tags,neighbors)
    # print(descriptor)

    # group_centers,group_indices,group_tags = build_index(items,tags)
    # bcutils.save_object([group_centers,group_indices,group_tags],"./model/group0802.obj")
    group_centers, group_indices, group_tags = bcutils.load_object("./model/group0802.obj")
    print("Build a total of", len(group_centers), "groups.")
    # group_algos = build_hnsw_index_second_level(items,group_indices)
    # bcutils.save_object(group_algos,"./model/group_algos_v1_0803.obj")
    group_algos = bcutils.load_object("./model/group_algos_v1_0803.obj")

    "Start query"

    num_group = 5
    num_candidate = num_group * 100
    candidate_groups, first_time = search_first_level(users, group_centers, group_indices, group_tags, num_group,
                                                      num_candidate)
    # print(candidate_groups[:10])

    results, second_time = search_second_level(items, users, candidate_groups, group_indices, group_algos,
                                               num_candidate)
    bcutils.save_object([results, second_time], "./model/excess_results_v500_0803.obj")
    # results,second_time = bcutils.load_object("./model/excess_results_v500_0803.obj")

    print(len(results), len(results[0]))
    print(results[0])
    print(np.percentile(np.array(results).reshape(-1), [0, 100]))
    # Consider 100 minimal select!!!
    results = transform_to_results(results)
    bcmetrics.evaluate(results, tags, neighbors)

# collective_candidates,cumulate_time = collect_candidates(items,users,candidate_groups,group_indices)

# print(len(collective_candidates),len(collective_candidates[0]),len(collective_candidates[1]))
# bcutils.save_object(collective_candidates,"./model/collective_candidates_v1_0803.obj")


"""


def test_descriptor(items,users,tags,neighbors,neighbor_count=100):
	descriptor,results = bcutils.load_object("results_hnsw_0802.obj")
	descriptor["diversity"] = bcmetrics.diversity(results, tags)
	descriptor["coverage"] = bcmetrics.coverage(results, neighbors)
	baseline_neighbors = transform_to_results(neighbors)
	descriptor["baseline_diversity"] = bcmetrics.diversity(baseline_neighbors, tags)
	descriptor["baseline_coverage"] = bcmetrics.coverage(baseline_neighbors, neighbors)
	random_neighbors = get_random_neighbors(items,users,neighbor_count)
	random_neighbors = transform_to_results(random_neighbors)
	descriptor["random_diversity"] = bcmetrics.diversity(random_neighbors, tags)
	descriptor["random_coverage"] = bcmetrics.coverage(random_neighbors, neighbors)
	return results,descriptor


"""
