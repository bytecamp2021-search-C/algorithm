import time

from lib.ann.ann_benchmarks.algorithms.definitions import (Definition, instantiate_algorithm)
from lib.ann.ann_benchmarks.distance import metrics
import numpy as np
from lib.ann.ann_benchmarks.runner import run_individual_query

from search.approximate_set_cover import approximate_set_cover, set_cover_plus
import bcmetrics
from utils import bcutils
from bcthread import MyThread

from collections import defaultdict


def collect_tag_index(items, tags, num_tags=100):
    member_indices = defaultdict(list)
    for i in range(len(tags)):
        for tag in tags[i]:
            member_indices[tag].append(i)
    group_indices = [np.array(member_indices[k]) for k in range(num_tags)]
    return group_indices


def get_random_neighbors(items, users, neighbor_count):
    results = []
    print("Start to random selection")
    t0 = time.time()
    index_set = np.arange(items.shape[0])
    for user in users:
        results.append(index_set[bcutils.random_sample(index_set, neighbor_count)])
    total_time = time.time() - t0
    print("Random selection in", total_time, "seconds")
    return np.array(results), total_time


def search_by_ann_benchmark(items, users, tags, neighbors, neighbor_count=100, random_ratio=None, evaluate=False):
    definition = Definition(
        algorithm="hnswlib",
        docker_tag=None,
        module="ann_benchmarks.algorithms.hnswlib",
        constructor="HnswLib",
        arguments=["angular", {"M": 8, "efConstruction": 500}],
        query_argument_groups=[[200]],
        disabled=False
    )
    algo = instantiate_algorithm(definition)
    distance = "angular"
    run_count = 1

    num_candidate = neighbor_count
    if random_ratio is not None:
        neighbor_count = int(num_candidate * random_ratio)

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
        if not query_argument_groups:
            query_argument_groups = [[]]
        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." %
                  (pos, len(query_argument_groups)))
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            t0 = time.time()
            descriptor, results = run_individual_query(
                algo, items, users, distance, neighbor_count, run_count, False)
            total_time = time.time() - t0
            print("Time Cost:", total_time)
            descriptor["build_time"] = build_time
            descriptor["query_time"] = total_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = definition.algorithm
            # bcutils.save_object([descriptor,results],"./model/results_hnsw_0802.obj")
            if random_ratio is not None:
                for i in range(len(results)):
                    select_idx = bcutils.random_sample(results[i][1], num_candidate)
                    results[i] = (results[i][0], [results[i][1][idx] for idx in select_idx])

            if evaluate == True:
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


def search_from_the_best(results, num_reserve=100):
    best_results = []
    for i in range(len(results)):
        candidates = sorted(results[i][1], key=lambda x: x[1])
        best_results.append([results[i][0], candidates[:num_reserve]])
    return best_results


def build_hnsw_index(items, group_indices):
    group_algos = []
    definition = Definition(
        algorithm="hnswlib",
        docker_tag=None,
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
        algo.fit(items[indices])
        build_time = time.time() - t0
        print('Built index for group #', group_idx, 'in', build_time, "seconds")
        query_arguments = definition.query_argument_groups[0]
        algo.set_query_arguments(*query_arguments)
        group_algos.append(algo)
    return group_algos


def single_query_hnsw(user, sub_items, count, algo, distance="angular"):
    start = time.time()
    candidates = algo.query(user, count)
    total = (time.time() - start)
    if distance:
        candidates = [[int(idx), float(metrics[distance]['distance'](user, sub_items[idx]))]
                      for idx in candidates]
    else:
        candidates = [[int(idx), 1.0] for idx in candidates]
    return candidates


def diversity_recall_single_thread(items, users, tags, group_indices, group_algos, tag_sets=None, diversity_coef=0.1,
                                   num_reserve=100):
    results = []
    num_recall = int(num_reserve * diversity_coef)
    total_time = 0
    for user_id in range(len(users)):
        t0 = time.time()
        candidates = dict()
        tag_set = tag_sets[user_id]

        num_sample_group = 10
        group_idx_list = bcutils.random_sample(group_indices, num_sample_group)
        for group_idx in group_idx_list:
            algo = group_algos[group_idx]
            item_id = group_indices[group_idx]

            num_cnt = 50
            res = single_query_hnsw(users[user_id], items[item_id], num_cnt, algo)
            for e in res:
                candidates[item_id[e[0]]] = e[1]

        indices = np.array(list(candidates.keys()))
        tags_list = tags[indices]
        cost_list = [candidates[k] for k in indices]

        res = approximate_set_cover(num_recall, cost_list, tags_list, tag_set)
        results.append(sorted([indices[e] for e in res]))
        total_time += (time.time() - t0)
        if user_id % 100 == 0:
            print("Current User #", user_id, "  Time cost:", total_time)
        if user_id >= 1000:
            break

    print("Time Cost:", total_time)
    return results, total_time


def greedy_set_cover(num_recall, tags_list, tag_set):
    tag_sets = [set(tags.tolist()) for tags in tags_list]
    exist_tags = tag_set
    res = set()
    for i in range(num_recall):
        best_score = len(exist_tags)
        nxt = None
        for j in range(len(tags_list)):
            if j in res:
                continue
            score = len(exist_tags | tag_sets[j])
            if nxt is None or score > best_score:
                nxt = j
                best_score = score
        res.add(nxt)
        exist_tags |= tag_sets[nxt]
    return list(res)


def diversity_recall_in_each_thread(pos, items, users, tags, group_indices, group_algos, num_recall, tag_sets,
                                    num_sample_group=25, cost_fun="distance+", method_name="set_cover+"):
    results = []
    for user_id in range(pos[0], pos[1]):
        t0 = time.time()
        tag_set = tag_sets[user_id]
        candidates = dict()
        group_idx_list = bcutils.random_sample(tag_set, num_sample_group)
        for group_idx in group_idx_list:
            algo = group_algos[group_idx]
            item_id = group_indices[group_idx]
            num_cnt = 5
            res = single_query_hnsw(users[user_id], items[item_id], num_cnt, algo)
            for e in res:
                candidates[item_id[e[0]]] = e[1]

        indices = np.array(list(candidates.keys()))
        tags_list = tags[indices]
        if cost_fun == "equal":
            cost_list = [1.0 for k in indices]
        elif cost_fun == "distance":
            cost_list = [candidates[k] for k in indices]
        else:
            cost_list = [1.0 + candidates[k] for k in indices]

        if method_name == "random":
            res = indices[bcutils.random_sample(indices, num_recall)]
            results.append(sorted(res))
        else:
            if method_name == "set_cover":
                res = approximate_set_cover(num_recall, cost_list, tags_list, tag_set)
            elif method_name == "set_cover+":
                res = set_cover_plus(num_recall, cost_list, tags_list, tag_set)
            else:
                res = greedy_set_cover(num_recall, tags_list, tag_set)
            results.append([indices[e] for e in res])

        if user_id % 100 == 0:
            print(user_id, "  Time Cost:", time.time() - t0)
    return results


def diversity_recall(items, users, tags, group_indices, group_algos, tag_sets=None, num_recall=10):
    results = []
    total_time = 0
    if tag_sets is None:
        tag_sets = [set(list(np.arange(100))) for u in users]
    num_thread = 10
    threads = []
    datasize = int(len(users) // num_thread) + 1
    t0 = time.time()
    for k in range(num_thread):
        threads.append(MyThread(k, diversity_recall_in_each_thread, \
                                [k * datasize, min((k + 1) * datasize, len(users))], items, users, tags, group_indices,
                                group_algos, num_recall, tag_sets))
        threads[k].start()

    for k in range(num_thread):
        threads[k].join()
        results.extend(threads[k].results)

    total_time = time.time() - t0
    print("Time Cost:", total_time)
    return results, total_time


def diversity_coverage_merge(cov_results, div_results, div_coef, num_recall=100):
    num_reserve = num_recall
    div_len = int(num_reserve * div_coef)
    merged_results = []
    for i in range(len(cov_results)):
        cov_res = set([div_results[i][1][k][0] for k in range(div_len)])
        for e in cov_results[i][1]:
            if len(cov_res) >= num_reserve:
                break
            if e[0] not in cov_res:
                cov_res.add(e[0])
        merged_results.append(sorted(list(cov_res)))
    return np.array(merged_results)


def diversity_coverage_merge_boost(tags, results, div_results, div_coef, num_reserve=100):
    div_len = int(num_reserve * div_coef)
    merged_results = []
    for i in range(len(results)):
        total_res = set([div_results[i][1][k][0] for k in range(div_len)])
        candidates = sorted(results[i][1], key=lambda x: x[1])
        tag_list = np.array([tags[e] for e in total_res]).reshape(-1).tolist()
        tag_set = set(tag_list)
        for e in candidates:
            if len(total_res) >= num_reserve:
                break
            new_tag = set(tags[e[0]].tolist())
            if len(new_tag & tag_set) < len(new_tag):
                total_res.add(e[0])
                tag_set |= new_tag
        for e in candidates:
            if len(total_res) >= num_reserve:
                break
            if e[0] not in total_res:
                total_res.add(e[0])
        merged_results.append(sorted(list(total_res)))
    return np.array(merged_results)


def obtain_valid_tags(tags, cov_results, div_coef=0):
    cov_len = int(len(cov_results[0][1]) * (1 - div_coef))
    valid_tags = []
    for i in range(len(cov_results)):
        exist_tags = np.array([tags[cov_results[i][1][j][0]] for j in range(cov_len)])
        exist_tags = set(exist_tags.reshape(-1).tolist())
        all_tags = set(list(np.arange(100)))
        valid_tags.append(all_tags - exist_tags)
    print("Set len:", np.mean([len(e) for e in valid_tags]))
    return valid_tags
