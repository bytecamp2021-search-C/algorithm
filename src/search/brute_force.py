import argparse
import time

import bcmetrics
from utils import bcutils


def brute_force(items, users):
    total_time = 0
    results = []
    for i in range(len(users)):
        t0 = time.time()
        dis = [[j, bcutils.calc_dist(users[i], items[j])] for j in range(len(items))]
        res = sorted(dis, key=lambda x: x[1])
        cur_time = time.time() - t0
        results.append([cur_time, res])
        print("User #", i, "  Current Time:", cur_time)
        total_time += cur_time
    return results, total_time


if __name__ == "__main__":
    items, users, tags, neighbors = bcutils.load_input_data()
    results, total_time = brute_force(items, users)
    bcmetrics.evaluate(results, tags, neighbors)
