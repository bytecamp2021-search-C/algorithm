import pickle
import h5py.h5f
import numpy as np

from lib.ann.ann_benchmarks.datasets import get_dataset
from lib.ann.ann_benchmarks.distance import dataset_transform


def load_input_data(dataset="glove-25-angular"):
    # D, dimension = get_dataset(dataset)
    D = h5py.File(f'data/{dataset}.hdf5', 'r')
    print(D.keys())
    tags = np.array(D['tags'])
    # tags = []
    distance = D.attrs['distance']
    neighbors = np.array(D['neighbors'])
    items, users = dataset_transform(D)
    print("Load data complete!")
    return items, users, tags, neighbors


'''
def load_input_data(dataset="glove-25-angular"):
	D, dimension = get_dataset(dataset)
	tags = np.array(D['tags'])
	distance = D.attrs['distance']
	neighbors = np.array(D['neighbors'])
	items, users = dataset_transform(D)
	print("Load data complete!")
	return items,users,tags,neighbors
'''


def save_object(obj, filename):
    f = open(filename, 'wb+')
    pickle.dump(obj, f)
    f.close()


def load_object(filename):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


def random_sample(a, n):
    m = len(a)
    if n / len(a) < 0.05:
        idx = set()
        while len(idx) < n:
            j = np.random.randint(m)
            if j not in idx:
                idx.add(j)
        return sorted(list(idx))
    else:
        idx = np.arange(m)
        np.random.shuffle(idx)
        return idx[:n]


def calc_dist(x_, y_, metric="cos"):
    x = x_.reshape(-1)
    y = y_.reshape(-1)
    if metric == "cos":
        return 1.0 - np.dot(x, y) / np.sqrt(np.dot(x, x)) / np.sqrt(np.dot(y, y))
    elif metric == "l2":
        return np.sqrt(np.dot(x - y, x - y))
    else:
        print("[!] unrecognized metric.")
        return 0.0


def softmax_dict_values(x: dict):
    sval = sum(np.exp(list(x.values())))
    return {k: np.exp(x[k]) / sval for k in x}


def regular_dict_values(x: dict):
    sval = max(sum(list(x.values())), 1e-6)
    return {k: x[k] / sval for k in x}


def transform_format_to_evaluate(raw_results):
    '''
    Input format: [num_user x num_recall]
    Output format: [num_user x (time,[num_recall x 2])]
    '''
    return [[0, [(e, 0) for e in item]] for item in raw_results]


def transform_format_from_evaluate(raw_results):
    '''
    Input format: [num_user x (time,[num_recall x 2])]
    Output format: [num_user x num_recall]
    '''
    return [[e[0] for e in item[1]] for item in raw_results]
