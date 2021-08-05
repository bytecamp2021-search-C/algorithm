import argparse
import time
import numpy as np
import sys

sys.path.append('..')
sys.path.append('lib/ann')

import bcmetrics
from utils import bcutils
from model import *


def test_hnsw_baseline(items, users, tags, neighbors, num_recall=100):
    results, descriptor = search_by_ann_benchmark(items, users, tags, neighbors, num_recall)
    bcutils.save_object([results, descriptor], "./results/results_hnsw_baseline_0804.obj")
    print(descriptor)
    bcmetrics.evaluate(results, tags, neighbors)


def test_random_recall(items, users, tags, neighbors, num_recall=100):
    random_results, total_time = get_random_neighbors(items, users, num_recall)
    random_results = bcutils.transform_format_to_evaluate(random_results)
    bcutils.save_object([random_results, total_time], "./results/results_random_recall_0804.obj")
    print("query_time:", total_time)
    bcmetrics.evaluate(random_results, tags, neighbors)


def test_hnsw_random_merge(items, users, tags, neighbors, num_recall=100):
    hnsw_results, descriptor = bcutils.load_object("./results/results_hnsw_baseline_0804.obj")
    random_results, random_time = bcutils.load_object("./results/results_random_recall_0804.obj")
    build_time = descriptor["build_time"]
    merge_time = 0
    eval_results = {"diversity_coef": [], "coverage": [], "diversity": []}
    div_coef_list = np.arange(0.0, 1.01, 0.1)
    for diversity_coef in div_coef_list:
        t0 = time.time()
        merged_results = diversity_coverage_merge(hnsw_results, random_results, diversity_coef)
        merge_time += time.time() - t0
        merged_results = bcutils.transform_format_to_evaluate(merged_results)
        res = bcmetrics.evaluate(merged_results, tags, neighbors)
        eval_results["diversity_coef"].append(diversity_coef)
        eval_results["coverage"].append(res["coverage"])
        eval_results["diversity"].append(res["diversity"])
    total_time = descriptor["query_time"] + random_time + (merge_time / len(div_coef_list))
    print(total_time)
    bcutils.save_object(eval_results, "./results/eval_results_hnsw_random_0804.obj")


def test_hnsw_plus_recall(items, users, tags, neighbors, num_recall=100):
    num_cnt_list = [200, 300, 400, 500, 600, 700, 800]
    for num_count in num_cnt_list:
        best_results, descriptor = search_by_ann_benchmark(items, users, tags, neighbors, num_count)
        bcutils.save_object([best_results, descriptor], "./results/results_hnsw+_best_%d_0804.obj" % num_count)
    for num_count in num_cnt_list:
        print(">> Num Count = ", num_count)
        best_results, descriptor = bcutils.load_object("./results/results_hnsw+_best_%d_0804.obj" % num_count)
        print(descriptor)
        top_results = search_from_the_best(best_results)
        res = bcmetrics.evaluate(top_results, tags, neighbors)
        random_results, random_time = bcutils.load_object("./results/results_random_recall_0804.obj")
        merged_results = diversity_coverage_merge(best_results, random_results, 0.1)
        merged_results = bcutils.transform_format_to_evaluate(merged_results)
        res = bcmetrics.evaluate(merged_results, tags, neighbors)


def test_build_index(items, tags):
    t0 = time.time()
    group_indices = collect_tag_index(items, tags)
    group_algos = build_hnsw_index(items, group_indices)
    total_time = time.time() - t0
    bcutils.save_object(group_algos, "./results/group_hnsw_index_0804.obj")
    print("Build index in", total_time, "seconds.")


def test_diversity_recall(items, users, tags, neighbors, num_recall=100):
    num_count = 500
    group_indices = collect_tag_index(items, tags)
    best_results, descriptor = bcutils.load_object("./results/results_hnsw+_best_%d_0804.obj" % num_count)
    tag_sets = obtain_valid_tags(tags, best_results, 0)
    group_algos = bcutils.load_object("./results/group_hnsw_index_0804.obj")
    # div_results,total_time = diversity_recall_single_thread(items,users,tags,group_indices,group_algos,tag_sets)
    num_select = 100
    div_results, total_time = diversity_recall(items, users, tags, group_indices, group_algos, tag_sets, num_select)
    div_results = bcutils.transform_format_to_evaluate(div_results)
    bcutils.save_object([div_results, total_time], "./results/results_diversity_recall_%d_0804.obj" % num_select)

    div_results, total_time = bcutils.load_object("./results/results_diversity_recall_%d_0804.obj" % num_select)
    merged_results = diversity_coverage_merge(best_results, div_results, 0.1)
    merged_results = bcutils.transform_format_to_evaluate(merged_results)
    bcutils.save_object(merged_results, "./results/results_merged_final_recall_0804.obj")
    res = bcmetrics.evaluate(merged_results, tags, neighbors)


def test_hnsw_plus_set_cover_model(items, users, tags, neighbors, num_recall=100):
    num_count = 500
    num_select = 100
    best_results, descriptor = bcutils.load_object("./results/results_hnsw+_best_%d_0804.obj" % num_count)
    div_results, total_time = bcutils.load_object("./results/results_diversity_recall_%d_0804.obj" % num_select)

    eval_results = [{"diversity_coef": [], "coverage": [], "diversity": []} for k in range(2)]
    ### coef & boost
    # div_coef_list = np.arange(0.0,0.201,0.01)
    div_coef_list = np.arange(0.0, 1.01, 0.1)
    for div_coef in div_coef_list:
        print(">> diversity coef:", div_coef)
        merged_results = diversity_coverage_merge(best_results, div_results, div_coef)
        merged_results = bcutils.transform_format_to_evaluate(merged_results)
        res = bcmetrics.evaluate(merged_results, tags, neighbors)
        eval_results[0]["diversity_coef"].append(div_coef)
        eval_results[0]["coverage"].append(res["coverage"])
        eval_results[0]["diversity"].append(res["diversity"])
        '''
        merged_results = diversity_coverage_merge_boost(tags,best_results,div_results,div_coef)
        merged_results = bcutils.transform_format_to_evaluate(merged_results)
        res = bcmetrics.evaluate(merged_results,tags,neighbors)
        eval_results[1]["diversity_coef"].append(div_coef)
        eval_results[1]["coverage"].append(res["coverage"])
        eval_results[1]["diversity"].append(res["diversity"])
        '''
    # bcutils.save_object(eval_results,"./results/eval_results_final_model_0804.obj")
    bcutils.save_object(eval_results[0], "./results/eval_results_hnsw+_setcover_0804.obj")


def test_avg_distance(items, users, tags, neighbors):
    ground_truth = bcutils.transform_format_to_evaluate(neighbors)
    hnsw_results, descriptor = bcutils.load_object("./results/results_hnsw_baseline_0804.obj")
    random_results, random_time = bcutils.load_object("./results/results_random_recall_0804.obj")
    merged_results_0 = diversity_coverage_merge(hnsw_results, random_results, 0.1)
    merged_results_0 = bcutils.transform_format_to_evaluate(merged_results_0)
    bcutils.save_object(merged_results_0, "./results/results_merged_hnsw_random_10%_0804.obj")
    merged_results_0 = bcutils.load_object("./results/results_merged_hnsw_random_10%_0804.obj")
    best_results, descriptor = bcutils.load_object("./results/results_hnsw+_best_500_0804.obj")
    top_results = search_from_the_best(best_results)
    merged_results = bcutils.load_object("./results/results_merged_final_recall_0804.obj")
    for res in [ground_truth, hnsw_results, random_results, merged_results_0, best_results, merged_results]:
        avg, std = bcmetrics.avg_distance(users, items, res, bcutils.calc_dist)
        print("avg_dist:", avg, "  std_dist:", std)


def subsampling_data(items, users, tags, neighbors, proportion=0.1, num_recall=100):
    item_idx = bcutils.random_sample(items, int(len(items) * proportion))
    idx_dict = {item_idx[k]: k for k in range(len(item_idx))}
    select_set = set(item_idx)
    sub_neighbors = []
    user_idx = []

    print(len(select_set))
    stats = []
    t0 = time.time()
    for user_id in range(len(users)):
        neighbor = []
        for e in neighbors[user_id]:
            if e in select_set:
                neighbor.append(idx_dict[e])
        stats.append(len(neighbor))
        if len(neighbor) < num_recall:
            continue
        user_idx.append(user_id)
        res = [[e, bcutils.calc_dist(users[user_id], items[e])] for e in neighbor]
        res = sorted(res, key=lambda x: x[1])
        sub_neighbors.append([res[i][0] for i in range(num_recall)])
        total_time = time.time() - t0
        if user_id % 100 == 0:
            print("Current user", user_id, "Total time:", total_time)
    print("Number of users reserved!", len(user_idx))
    print(user_idx[0])
    print(np.percentile(stats, [0, 25, 50, 75, 100]))
    return items[item_idx], users[user_idx], tags[item_idx], np.array(sub_neighbors)


if __name__ == "__main__":
    items, users, tags, neighbors = bcutils.load_input_data()
    # bcutils.save_object([items,users,tags,neighbors],"./data/dataset_subsample_10%%.obj")

    test_hnsw_baseline(items, users, tags, neighbors)
    test_random_recall(items, users, tags, neighbors)
    test_hnsw_random_merge(items, users, tags, neighbors)
    test_hnsw_plus_recall(items, users, tags, neighbors)
    test_build_index(items, tags)
    test_diversity_recall(items, users, tags, neighbors)
    test_hnsw_plus_set_cover_model(items, users, tags, neighbors)
    test_avg_distance(items, users, tags, neighbors)
