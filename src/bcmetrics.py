import numpy as np


def diversity(results, tags):
    tag_c = 0
    for _, items in results:
        tag_count = {}
        for item in items:
            tag = tags[item[0]]
            for e in tag:
                if e in tag_count.keys():
                    tag_count[e] = tag_count[e] + 1
                else:
                    tag_count[e] = 1
        tag_c += len(tag_count)
    return tag_c / len(results)


def coverage(results, neighbors):
    coverage = 0.0
    for i, candidate in enumerate(results):
        neighbor = neighbors[i]
        match = 0
        for item in candidate[1]:
            if item[0] in neighbor:
                match = match + 1
        coverage = coverage + match / len(neighbor)
    return coverage / len(results)


def evaluate(results, tags, neighbors):
    descriptor = dict()
    descriptor["diversity"] = diversity(results, tags)
    descriptor["coverage"] = coverage(results, neighbors)
    print(descriptor)
